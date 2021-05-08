import tensorflow as tf
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Pool2d


class StartRes2Conv(Layer):

    def __init__(self, channels, kernel_size, stride=1, dilation=1, scale=4, groups=1, norm='def', act='def'):
        super().__init__()
        assert channels % scale == 0
        width = channels // scale
        self.direct = Pool2d(3, 2, type='avg') if stride == 2 else Identity()
        self.conv = Conv2d(width * (scale - 1), width * (scale - 1),
                           kernel_size=kernel_size, stride=stride, dilation=dilation,
                           groups=groups * (scale - 1), norm=norm, act=act)
        self.width = width

    def call(self, x):
        xs = tf.split(x, [self.width, -1], axis=-1)
        outs = [self.direct(xs[0]), self.conv(xs[1])]
        x = tf.concat(outs, axis=-1)
        return x


class StartRes2Conv2(Layer):

    def __init__(self, channels, kernel_size, stride=1, dilation=1, scale=4, groups=1, norm='def', act='def'):
        super().__init__()
        assert channels % scale == 0
        width = channels // scale
        self.scale = scale
        self.direct = Pool2d(3, 2, type='avg') if stride == 2 else Identity()
        self.convs = [
            Conv2d(width, width, kernel_size=kernel_size, dilation=dilation,
                   stride=stride, groups=groups, norm=norm, act=act)
            for i in range(scale - 1)
        ]

    def call(self, x):
        xs = tf.split(x, self.scale, axis=-1)
        outs = [self.direct(xs[0]), self.convs[0](xs[1])]
        for i in range(2, self.scale):
            x = xs[i]
            x = self.convs[i - 1](x)
            outs.append(x)
        x = tf.concat(outs, axis=-1)
        return x


class Res2Conv(Layer):

    def __init__(self, channels, kernel_size, stride=1, dilation=1, scale=4, groups=1, norm='def', act='def',
                 start_block=True):
        super().__init__()
        assert channels % scale == 0
        width = channels // scale
        self.scale = scale
        self.
        self.direct = Pool2d(3, 2, type='avg') if stride == 2 else Identity()
        self.convs = [
            Conv2d(width, width, kernel_size=kernel_size, dilation=dilation,
                   groups=groups, norm=norm, act=act)
            for i in range(scale - 1)
        ]

    def call(self, x):
        xs = tf.split(x, self.scale, axis=-1)
        outs = [
            xs[0], self.convs[0](xs[1])
        ]
        for i in range(2, self.scale):
            x = outs[-1] + xs[i]
            x = self.convs[i - 1](x)
            outs.append(x)
        x = tf.concat(outs, axis=-1)
        return x


class MainRes2Conv(Layer):

    def __init__(self, channels, kernel_size, dilation=1, scale=4, groups=1, norm='def', act='def'):
        super().__init__()
        assert channels % scale == 0
        width = channels // scale
        self.scale = scale
        self.convs = [
            Conv2d(width, width, kernel_size=kernel_size, dilation=dilation,
                   groups=groups, norm=norm, act=act)
            for i in range(scale - 1)
        ]

    def call(self, x):
        xs = tf.split(x, self.scale, axis=-1)
        outs = [
            xs[0], self.convs[0](xs[1])
        ]
        for i in range(2, self.scale):
            x = outs[-1] + xs[i]
            x = self.convs[i - 1](x)
            outs.append(x)
        x = tf.concat(outs, axis=-1)
        return x


def Res2Conv(in_channels, out_channels, kernel_size, stride, dilation, scale, groups, norm, act, start_block):
    assert in_channels == out_channels
    if scale == 1:
        return Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      dilation=dilation, groups=groups, norm=norm, act=act)
    if start_block:
        return StartRes2Conv(in_channels, kernel_size, stride, dilation, scale, groups, norm, act)
    else:
        return MainRes2Conv(in_channels, kernel_size, dilation, scale, groups, norm, act)
