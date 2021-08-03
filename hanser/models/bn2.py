import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers


class BatchNormalizationTest(Layer):
    r"""Layer that normalizes its inputs.

    Batch normalization applies a transformation that maintains the mean output
    close to 0 and the output standard deviation close to 1.

    Importantly, batch normalization works differently during training and
    during inference.

    **During training** (i.e. when using `fit()` or when calling the layer/model
    with the argument `training=True`), the layer normalizes its output using
    the mean and standard deviation of the current batch of inputs. That is to
    say, for each channel being normalized, the layer returns
    `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

    - `epsilon` is small constant (configurable as part of the constructor
    arguments)
    - `gamma` is a learned scaling factor (initialized as 1), which
    can be disabled by passing `scale=False` to the constructor.
    - `beta` is a learned offset factor (initialized as 0), which
    can be disabled by passing `center=False` to the constructor.

    **During inference** (i.e. when using `evaluate()` or `predict()` or when
    calling the layer/model with the argument `training=False` (which is the
    default), the layer normalizes its output using a moving average of the
    mean and standard deviation of the batches it has seen during training. That
    is to say, it returns
    `gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta`.

    `self.moving_mean` and `self.moving_var` are non-trainable variables that
    are updated each time the layer in called in training mode, as such:

    - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
    - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

    As such, the layer will only normalize its inputs during inference
    *after having been trained on data that has similar statistics as the
    inference data*.

    Args:
      axis: Integer or a list of integers, the axis that should be normalized
      (typically the features axis). For instance, after a `Conv2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used. When the
        next layer is linear (also e.g. `nn.relu`), this can be disabled since the
        scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      trainable: Boolean, if `True` the variables will be marked as trainable.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        - `training=True`: The layer will normalize its inputs using the mean and
          variance of the current batch of inputs.
        - `training=False`: The layer will normalize its inputs using the mean and
          variance of its moving statistics, learned during training.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.

    Output shape:
      Same shape as input.

    Reference:
      - [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).
    """

    # By default, the base class uses V2 behavior. The BatchNormalization V1
    # subclass sets this to False to use the V1 behavior.
    _USE_V2_BEHAVIOR = True

    def __init__(self,
                 axis=-1,
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
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)
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
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank.')
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %s' % (self.axis,))
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % (self.axis,))

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Received input '
                                 'with shape %s. Axis value: %s' %
                                 (tuple(input_shape), self.axis))
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
            ]

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
    def _assign_moving_average(self, variable, value, momentum, inputs_size):

        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(inputs_size > 0, update_delta,
                                        tf.zeros_like(update_delta))
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

        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)

        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        # noinspection PyUnresolvedReferences
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        if training:
            # noinspection PyTypeChecker
            keep_dims = len(self.axis) > 1
            mean, variance = tf.nn.moments(
                tf.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            new_mean, new_variance = mean, variance

            def _do_update(var, value):
                return self._assign_moving_average(var, value, self.momentum, input_batch_size=None)

            def mean_update():
                return _do_update(self.moving_mean, new_mean)

            def variance_update():
                 return _do_update(self.moving_variance, new_variance)

            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean, variance = self.moving_mean, self.moving_variance

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(inputs, _broadcast(mean),
                                            _broadcast(variance), offset, scale,
                                            self.epsilon)
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
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
