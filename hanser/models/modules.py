import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, InputSpec, Softmax, Dropout

from hanser.models.layers import Conv2d, Norm, Act


class PadChannel(Layer):

    def __init__(self, c, name=None):
        super().__init__(name=name)
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

    def __init__(self, in_channels, reduction, groups=1, name=None):
        super().__init__(name=name)
        channels = in_channels // reduction
        self.pool = GlobalAvgPool(keep_dim=True, name='pool')
        self.fc = Sequential([
            Conv2d(in_channels, channels, 1, groups=groups, act='relu', name='fc1'),
            Conv2d(channels, in_channels, 1, groups=groups, act='sigmoid', name='fc2'),
        ])

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


class DropPath(Dropout):

    def __init__(self, rate, **kwargs):
        super().__init__(rate, noise_shape=(None, 1, 1, 1), **kwargs)


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
        super().__init__(name=name)
        inter_channels = min(max(in_channels * radix // reduction, 32), in_channels)
        inter_channels = round_channels(inter_channels, groups * radix)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, groups=groups * radix,
                           dilation=dilation, bias=bias, name=name + "/conv")
        self.bn = Norm(channels * radix, name=name + "/bn")
        self.act = Act(name=name + "/act")
        self.attn = Sequential([
            GlobalAvgPool(keep_dim=True, name=name + "/attn/pool"),
            Conv2d(channels, inter_channels, 1, groups=groups, norm='default', act='default', name=name + "/attn/fc1"),
            Conv2d(inter_channels, channels * radix, 1, groups=groups, name=name + "/attn/fc2"),
            rSoftMax(radix, groups, name=name + "/attn/rsoftmax"),
        ], name=name + "/attn")

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
