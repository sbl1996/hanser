import tensorflow as tf
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Pool2d


class Res2Conv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, scale=4, groups=1,
                 norm='def', act='def', start_block=True):
        super().__init__()
        assert in_channels == out_channels
        channels = in_channels
        assert channels % scale == 0
        width = channels // scale
        self.scale = scale
        self.start_block = start_block
        self.direct = Pool2d(3, 2, type='avg') if stride == 2 else Identity()
        self.convs = [
            Conv2d(width, width, kernel_size=kernel_size, stride=stride,
                   dilation=dilation, groups=groups, norm=norm, act=act)
            for _i in range(scale - 1)
        ]

    def call(self, x):
        xs = tf.split(x, self.scale, axis=-1)
        outs = [
            self.direct(xs[0]), self.convs[0](xs[1])
        ]
        for i in range(2, self.scale):
            x = xs[i]
            if not self.start_block:
                x = outs[-1] + xs[i]
            x = self.convs[i - 1](x)
            outs.append(x)
        x = tf.concat(outs, axis=-1)
        return x
