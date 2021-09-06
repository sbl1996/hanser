import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers


def _remove_bessels_correction(sample_size, variance):
    sample_size = tf.cast(sample_size, variance.dtype)
    factor = (sample_size - tf.cast(1.0, variance.dtype)) / sample_size
    return variance * factor

# From tf.keras.layers.experimental.SyncBatchNormalization
def global_moments(x, axes=(0, 1, 2), keepdims=False):
    y = tf.cast(x, tf.float32) if x.dtype == tf.float16 else x
    replica_ctx = tf.distribute.get_replica_context()
    if replica_ctx:
        shape = tf.shape(y)
        local_sum = tf.reduce_sum(y, axis=axes, keepdims=True)
        local_squared_sum = tf.reduce_sum(tf.square(y), axis=axes, keepdims=True)
        batch_size = tf.cast(shape[axes[0]], tf.float32)
        y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
        y_squared_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_squared_sum)
        global_batch_size = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, batch_size)

        axes_vals = [shape[axes[i]] for i in range(1, len(axes))]
        multiplier = tf.cast(tf.reduce_prod(axes_vals), tf.float32)
        multiplier = multiplier * global_batch_size

        mean = y_sum / multiplier
        y_squared_mean = y_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - tf.square(mean)
    else:
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    if not keepdims:
        mean = tf.squeeze(mean, axes)
        variance = tf.squeeze(variance, axes)
    if x.dtype == tf.float16:
        return tf.cast(mean, tf.float16), tf.cast(variance, tf.float16)
    else:
        return mean, variance


def make_inplace_abn_op(epsilon, alpha, fused=True, sync=False):
    if sync: assert not fused

    @tf.custom_gradient
    def InplaceABNOp(x, scale, offset):

        gamma, beta = scale, offset
        if fused:
            y, mean, var, _, _, _ = tf.raw_ops.FusedBatchNormV3(
                x=x, scale=gamma, offset=beta, mean=tf.constant([]), variance=tf.constant([]),
                epsilon=epsilon, exponential_avg_factor=1, is_training=True)

            shape = tf.shape(x)
            sample_size = tf.cast(tf.reduce_prod([shape[i] for i in (0, 1, 2)]), x.dtype)
            var = _remove_bessels_correction(sample_size, var)
        else:
            if sync:
                mean, var = global_moments(x, axes=(0, 1, 2), keepdims=False)
            else:
                mean, var = tf.nn.moments(x, axes=(0, 1, 2), keepdims=False)
            ginv = tf.math.rsqrt(var + epsilon) * gamma
            y = x * tf.cast(ginv, x.dtype) + tf.cast(beta - mean * ginv, x.dtype)

        z = tf.nn.leaky_relu(y, alpha)
        def custom_grad(dz, _d1, _d2):
            # depends on z, var, gamma, beta
            shape = tf.shape(dz)
            m = tf.cast(shape[0] * shape[1] * shape[2], dz.dtype)
            mask = z >= 0
            y = tf.where(mask, z, z / alpha)
            dy = tf.where(mask, 1.0, alpha) * dz

            dbeta = tf.reduce_sum(dy, axis=(0, 1, 2))

            x_ = (y - beta) / gamma
            dgamma = tf.reduce_sum(dy * x_, axis=(0, 1, 2))
            ginv = gamma * tf.math.rsqrt(var + epsilon)
            dx = (dy - dgamma / m * x_ - dbeta / m) * ginv

            # TODO: Inplace ABN-II
            # Maybe computationally more efﬁcient, but encounter numeric instability
            # dgamma = (tf.reduce_sum(dy * y, axis=(0, 1, 2)) - beta * dbeta) / gamma
            # ginv = gamma * tf.math.rsqrt(var + epsilon)
            # d_t = dgamma / gamma
            # dx = (dy - d_t * y / m - (dbeta - beta * d_t) / m) * ginv
            return dx, dgamma, dbeta
        return (z, mean, var), custom_grad
    return InplaceABNOp


class InplaceABN(Layer):

    def __init__(self,
                 momentum=0.9,
                 epsilon=1.001e-5,
                 center=True,
                 scale=True,
                 alpha=0.01,
                 fused=None,
                 sync=False,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if fused is None:
            fused = not sync
        if sync:
            assert not fused
        self.axis = -1
        self.momentum = momentum
        # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
        # prevent exception (see cudnn.h).
        min_epsilon = 1.001e-5
        epsilon = min_epsilon if epsilon < min_epsilon else epsilon
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.alpha = alpha
        self.fused = fused
        self.sync = sync
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.supports_masking = True

        self.trainable = trainable

        self._op_func = make_inplace_abn_op(self.epsilon, self.alpha, fused=self.fused, sync=True)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if self.axis < 0:
            self.axis = 4 + self.axis

        axis_to_dim = {self.axis: input_shape.dims[self.axis].value}
        if axis_to_dim[self.axis] is None:
            raise ValueError('Input has undefined `axis` dimension. Received input '
                             'with shape %s. Axis value: %s' %
                             (tuple(input_shape), self.axis))
        self.input_spec = InputSpec(ndim=4, axes=axis_to_dim)

        param_shape = (list(axis_to_dim.values())[0],)

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False)

            self.moving_variance = self.add_weight(
                name='moving_variance',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False)
        finally:
            # noinspection PyUnboundLocalVariable
            if partitioner:
                # noinspection PyUnboundLocalVariable
                self._scope.set_partitioner(partitioner)
        self.built = True

    # noinspection PyMethodMayBeStatic
    def _assign_moving_average(self, variable, value, momentum):

        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            return update_delta

        with tf.name_scope('AssignMovingAvg') as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope)

    # noinspection PyMethodOverriding
    def call(self, inputs, training=None):
        input_shape = inputs.shape
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)

        if training:
            scale, offset = self.gamma, self.beta
            if offset is not None:
                offset = tf.cast(offset, inputs.dtype)
            if scale is not None:
                scale = tf.cast(scale, inputs.dtype)
            output, mean, variance = self._op_func(inputs, scale, offset)

            def _do_update(var, value):
                return self._assign_moving_average(var, value, self.momentum)

            def mean_update():
                return _do_update(self.moving_mean, mean)

            def variance_update():
                 return _do_update(self.moving_variance, variance)

            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean, variance = self.moving_mean, self.moving_variance
            scale, offset = self.gamma, self.beta
            mean = tf.cast(mean, inputs.dtype)
            variance = tf.cast(variance, inputs.dtype)
            if offset is not None:
                offset = tf.cast(offset, inputs.dtype)
            if scale is not None:
                scale = tf.cast(scale, inputs.dtype)

            output, _mean, _var = tf.compat.v1.nn.fused_batch_norm(
                inputs, scale, offset, mean, variance, self.epsilon, is_training=False)
            output = tf.nn.leaky_relu(output, alpha=self.alpha)

        if inputs_dtype in (tf.float16, tf.bfloat16):
            output = tf.cast(output, inputs_dtype)

        output.set_shape(input_shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'alpha': self.alpha,
            'fused': self.fused,
            'sync': self.sync,
            'beta_initializer':
                initializers.serialize(self.beta_initializer),
            'gamma_initializer':
                initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer':
                regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer':
                regularizers.serialize(self.gamma_regularizer),
        }