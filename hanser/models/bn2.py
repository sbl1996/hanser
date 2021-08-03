import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers


class BatchNormalizationTest(Layer):

    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
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
        self.axis = -1
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.supports_masking = True

        self.trainable = trainable

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
            mean, variance = tf.nn.moments(
                tf.cast(inputs, self._param_dtype), [0, 1, 2])

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
        outputs = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, self.epsilon)

        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        outputs.set_shape(input_shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'axis':
                self.axis,
            'momentum':
                self.momentum,
            'epsilon':
                self.epsilon,
            'center':
                self.center,
            'scale':
                self.scale,
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
