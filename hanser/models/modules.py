import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputSpec, Softmax, Dropout, Layer
from tensorflow.python.keras.utils.control_flow_util import smart_cond

from hanser.models.layers import Conv2d, Norm, Act


class PadChannel(Layer):

    def __init__(self, c, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def call(self, x, training=None):
        return tf.pad(x, [(0, 0), (0, 0), (0, 0), (0, self.c)])

    def compute_output_shape(self, input_shape):
        in_channels = input_shape[-1]
        return input_shape[:-1] + (in_channels + self.c,)

    def get_config(self):
        config = {'c': self.c}
        base_config = super().get_config()
        return {**base_config, **config}


class SELayer(Layer):

    def __init__(self, in_channels, reduction, groups=1, **kwargs):
        super().__init__(**kwargs)
        channels = min(max(in_channels // reduction, 32), in_channels)
        if groups != 1:
            channels = round_channels(channels, groups)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.fc = Sequential([
            Conv2d(in_channels, channels, 1, groups=groups, act='def'),
            Conv2d(channels, in_channels, 1, groups=groups, act='sigmoid'),
        ])

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


class DropPath(Dropout):

    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = self.add_weight(
            name="drop_rate", shape=(), dtype=tf.float32,
            initializer=initializers.Constant(rate), trainable=False,
        )

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        noise_shape = (tf.shape(inputs)[0], 1, 1, 1)

        def dropped_inputs():
            return tf.nn.dropout(
                inputs,
                noise_shape=noise_shape,
                rate=self.rate)

        output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
        }
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rSoftMax(Layer):

    def __init__(self, radix, cardinality, **kwargs):
        super().__init__(**kwargs)
        self.radix = radix
        self.cardinality = cardinality
        self.input_spec = InputSpec(ndim=4)
        if self.radix > 1:
            self.softmax = Softmax(axis=1, dtype='float32')

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def call(self, x):
        if self.radix > 1:
            b = tf.shape(x)[0]
            c = x.shape[-1]
            ic = c // self.cardinality // self.radix
            x = tf.reshape(x, [b, self.cardinality, self.radix, ic])
            x = tf.transpose(x, [0, 2, 1, 3])
            xs = self.softmax(x)
            if xs.dtype != x.dtype:
                x = tf.cast(xs, x.dtype)
            else:
                x = xs
            # x = tf.nn.softmax(x, axis=1)
            x = tf.reshape(x, [b, 1, 1, c])
        else:
            x = tf.sigmoid(x)
        return x

    def get_config(self):
        config = {'radix': self.radix, 'cardinality': self.cardinality}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class SplAtConv2d(Layer):

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding='same',
                 dilation=1, groups=1, bias=None, radix=2, reduction=4, name=None):
        super().__init__()
        inter_channels = min(max(in_channels * radix // reduction, 32), in_channels)
        inter_channels = round_channels(inter_channels, groups * radix)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, groups=groups * radix,
                           dilation=dilation, bias=bias)
        self.bn = Norm(channels * radix)
        self.act = Act()
        self.attn = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(channels, inter_channels, 1, groups=groups, norm='default', act='default'),
            Conv2d(inter_channels, channels * radix, 1, groups=groups),
            rSoftMax(radix, groups),
        ])

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        if self.radix > 1:
            splited = tf.split(x, self.radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        gap = self.attn(gap)

        if self.radix > 1:
            attns = tf.split(gap, self.radix, axis=-1)
            out = sum([attn * split for (attn, split) in zip(attns, splited)])
        else:
            out = gap * x
        return out


class GlobalAvgPool(Layer):
    """Abstract class for different global pooling 2D layers.
  """

    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim
        self.input_spec = InputSpec(ndim=4)
        self._supports_ragged_inputs = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.keep_dim:
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])
        else:
            return tf.TensorShape([input_shape[0], input_shape[3]])

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=self.keep_dim)

    def get_config(self):
        config = {'keep_dim': self.keep_dim}
        base_config = super().get_config()
        return {**base_config, **config}


class ReZero(Layer):

    def __init__(self, init_val=0., **kwargs):
        super().__init__(**kwargs)
        self.init_val = init_val
        self.res_weight = self.add_weight(
            name='res_weight', shape=(), dtype=tf.float32,
            trainable=True, initializer=initializers.Constant(init_val))

    def call(self, x):
        return x * self.res_weight

    def get_config(self):
        base_config = super(ReZero, self).get_config()
        return base_config


class Affine(Layer):

    def __init__(self,
                 axis=-1,
                 scale=True,
                 center=True,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)
        self.scale = scale
        self.center = center
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)

    def call(self, inputs):
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1)):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        if scale is not None:
            inputs = inputs * scale

        if offset is not None:
            inputs = inputs + offset
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, inputs_dtype)
        return inputs

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
            raise ValueError('Input has undefined rank:', input_shape)
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
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                 input_shape)
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
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None

    def get_config(self):
        base_config = super(Affine, self).get_config()
        base_config = {
            "scale": self.scale,
            "center": self.center,
            **base_config
        }
        return base_config
