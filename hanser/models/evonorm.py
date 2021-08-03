import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers

# def evonorm_s0(inputs,
#             is_training,
#             nonlinearity=True,
#             init_zero=False,
#             decay=0.9,
#             epsilon=1e-5,
#             num_groups=32):
#     if init_zero:
#         gamma_initializer = tf.zeros_initializer()
#     else:
#         gamma_initializer = tf.ones_initializer()
#
#     var_shape = (1, 1, 1, inputs.shape[3])
#     with tf.variable_scope(None, default_name='evonorm'):
#         beta = tf.get_variable(
#             'beta',
#             shape=var_shape,
#             dtype=inputs.dtype,
#             initializer=tf.zeros_initializer())
#         gamma = tf.get_variable(
#             'gamma',
#             shape=var_shape,
#             dtype=inputs.dtype,
#             initializer=gamma_initializer)
#         if nonlinearity:
#             v = tf.get_variable(
#                 'v',
#                 shape=var_shape,
#                 dtype=inputs.dtype,
#                 initializer=tf.ones_initializer())
#             den = _group_std(
#                 inputs,
#                 epsilon=epsilon,
#                 num_groups=num_groups)
#             inputs = inputs * tf.nn.sigmoid(v * inputs) / den
#
#     return inputs * gamma + beta
#
#
# def evonorm_b0(inputs, is_training, nonlinearity=True, init_zero=False, decay=0.9, epsilon=1e-5):
#
#     if init_zero:
#         gamma_initializer = tf.zeros_initializer()
#     else:
#         gamma_initializer = tf.ones_initializer()
#
#     var_shape = (1, 1, 1, inputs.shape[3])
#     with tf.variable_scope(None, default_name='evonorm'):
#         beta = tf.get_variable(
#             'beta',
#             shape=var_shape,
#             dtype=inputs.dtype,
#             initializer=tf.zeros_initializer())
#         gamma = tf.get_variable(
#             'gamma',
#             shape=var_shape,
#             dtype=inputs.dtype,
#             initializer=gamma_initializer)
#         if nonlinearity:
#             v = tf.get_variable(
#                 'v',
#                 shape=var_shape,
#                 dtype=inputs.dtype,
#                 initializer=tf.ones_initializer())
#             left = _batch_std(
#                 inputs,
#                 decay=decay,
#                 epsilon=epsilon,
#                 training=is_training)
#             right = v * inputs + _instance_std(inputs, epsilon=epsilon)
#             inputs = inputs / tf.maximum(left, right)
#     return inputs * gamma + beta
#
#
# def _batch_std(inputs, training, decay=0.9, epsilon=1e-5):
#     """Batch standard deviation."""
#     var_shape, axes = (1, 1, 1, inputs.shape[3]), [0, 1, 2]
#     moving_variance = tf.get_variable(
#         name="moving_variance",
#         shape=var_shape,
#         initializer=tf.initializers.ones(),
#         dtype=tf.float32,
#         collections=[
#             tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
#             tf.GraphKeys.GLOBAL_VARIABLES
#         ],
#         trainable=False)
#     if training:
#         _, variance = tf.nn.moments(inputs, axes, keep_dims=True)
#         variance = tf.cast(variance, tf.float32)
#         update_op = tf.assign_sub(
#             moving_variance,
#             (moving_variance - variance) * (1 - decay))
#         tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
#     else:
#         variance = moving_variance
#     std = tf.sqrt(variance + epsilon)
#     return tf.cast(std, inputs.dtype)


def _instance_std(inputs, epsilon=1e-5):
    """Instance standard deviation."""
    axes = [1, 2]
    _, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
    return tf.sqrt(variance + epsilon)


def _group_std(inputs,
               epsilon=1e-5,
               num_groups=32):
    """Grouped standard deviation along the channel dimension."""
    axis = 3
    while num_groups > 1:
        if inputs.shape[axis] % num_groups == 0:
            break
        num_groups -= 1
    _, h, w, c = inputs.shape.as_list()
    x = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    _, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
    std = tf.sqrt(variance + epsilon)
    std = tf.broadcast_to(std, _get_shape_list(x))
    return tf.reshape(std, _get_shape_list(inputs))


def _get_shape_list(tensor):
    """Returns tensor's shape as a list which can be unpacked."""
    static_shape = tensor.shape.as_list()
    if not any([x is None for x in static_shape]):
        return static_shape

    dynamic_shape = tf.shape(tensor)
    ndims = tensor.shape.ndims

    # Return mixture of static and dynamic dims.
    shapes = [
        static_shape[i] if static_shape[i] is not None else dynamic_shape[i]
        for i in range(ndims)
    ]
    return shapes


class EvoNormB0(Layer):

    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 v_initializer='ones',
                 moving_variance_initializer='ones',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = -1
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.v_initializer = initializers.get(v_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)

        self.trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def _param_dtype(self):
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

        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.gamma_initializer,
            trainable=True,
            experimental_autocast=False)
        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.beta_initializer,
            trainable=True,
            experimental_autocast=False)
        self.v = self.add_weight(
            name='v',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.v_initializer,
            trainable=True,
            experimental_autocast=False)

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None

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
            _mean, variance = tf.nn.moments(
                tf.cast(inputs, self._param_dtype), [0, 1, 2])

            def _do_update(var, value):
                return self._assign_moving_average(var, value, self.momentum)

            def variance_update():
                 return _do_update(self.moving_variance, variance)

            self.add_update(variance_update)
        else:
            variance = self.moving_variance

        variance = tf.cast(variance, inputs.dtype)
        offset = tf.cast(self.beta, inputs.dtype)
        scale = tf.cast(self.gamma, inputs.dtype)
        v = tf.cast(self.v, inputs.dtype)

        left = tf.sqrt(variance + self.epsilon)
        right = v * inputs + _instance_std(inputs, epsilon=self.epsilon)
        inputs = inputs / tf.maximum(left, right)
        outputs = inputs * scale + offset

        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        outputs.set_shape(input_shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'momentum':
                self.momentum,
            'epsilon':
                self.epsilon,
            'beta_initializer':
                initializers.serialize(self.beta_initializer),
            'gamma_initializer':
                initializers.serialize(self.gamma_initializer),
            'v_initializer':
                initializers.serialize(self.v_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
        }


class EvoNormS0(Layer):

    def __init__(self,
                 num_groups=32,
                 momentum=0.9,
                 epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 v_initializer='ones',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = -1
        self.num_groups = num_groups
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.v_initializer = initializers.get(v_initializer)

        self.trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def _param_dtype(self):
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

        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.gamma_initializer,
            trainable=True,
            experimental_autocast=False)
        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.beta_initializer,
            trainable=True,
            experimental_autocast=False)
        self.v = self.add_weight(
            name='v',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.v_initializer,
            trainable=True,
            experimental_autocast=False)
        self.built = True

    # noinspection PyMethodOverriding
    def call(self, inputs, training=None):
        input_shape = inputs.shape
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)


        offset = tf.cast(self.beta, inputs.dtype)
        scale = tf.cast(self.gamma, inputs.dtype)
        v = tf.cast(self.v, inputs.dtype)
        den = _group_std(inputs, self.epsilon, self.num_groups)

        inputs = inputs * tf.nn.sigmoid(v * inputs) / den
        outputs = inputs * scale + offset

        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        outputs.set_shape(input_shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_groups':
                self.num_groups,
            'momentum':
                self.momentum,
            'epsilon':
                self.epsilon,
            'beta_initializer':
                initializers.serialize(self.beta_initializer),
            'gamma_initializer':
                initializers.serialize(self.gamma_initializer),
            'v_initializer':
                initializers.serialize(self.v_initializer),
        }
