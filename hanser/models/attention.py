import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Softmax, InputSpec

from hanser.models.layers import GlobalAvgPool, Conv2d, Norm, Act


class SELayer(Layer):

    def __init__(self, in_channels, reduction=None, groups=1, se_channels=None,
                 min_se_channels=32, act='def', mode=0, **kwargs):
        super().__init__(**kwargs)
        self.pool = GlobalAvgPool(keep_dim=True)
        if mode == 0:
            # σ(f_{W1, W2}(y))
            channels = se_channels or min(max(in_channels // reduction, min_se_channels), in_channels)
            if groups != 1:
                channels = round_channels(channels, groups)
            self.fc = Sequential([
                Conv2d(in_channels, channels, kernel_size=1, bias=False, act=act),
                Conv2d(channels, in_channels, 1, groups=groups, act='sigmoid'),
            ])
        elif mode == 1:
            # σ(w ⊙ y)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1,
                             groups=in_channels, bias=False, act='sigmoid')
        elif mode == 2:
            # σ(Wy)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1, bias=False, act='sigmoid')
        else:
            raise ValueError("Not supported mode: {}" % mode)

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


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