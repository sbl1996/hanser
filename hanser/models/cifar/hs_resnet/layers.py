import tensorflow as tf
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d

def split2(x):
    c = x.shape[-1]
    return tf.split(x, [c - c // 2, c // 2], axis=-1)


class MainHSConv(Layer):

    def __init__(self, channels, kernel_size, groups=1, split=5, norm='def', act='def'):
        super().__init__()
        assert channels % split == 0
        width = channels // split
        self.split = split

        c = width
        self.convs = []
        for i in range(split - 1):
            self.convs.append(
                Conv2d(c, c, kernel_size=kernel_size, groups=groups, norm=norm, act=act)
            )
            c = width + c // 2

    def call(self, x):
        xs = tf.split(x, self.split, axis=-1)
        outs = [xs[0]]
        x = xs[1]
        for i in range(1, self.split - 1):
            x = self.convs[i - 1](x)
            x1, x2 = split2(x)
            outs.append(x1)
            x = tf.concat((xs[i + 1], x2), axis=-1)
        outs.append(self.convs[-1](x))
        x = tf.concat(outs, axis=-1)
        return x

def HSConv(in_channels, out_channels, kernel_size, stride=1, groups=1, split=5, norm='def', act='def'):
    assert in_channels == out_channels
    if stride == 1:
        return MainHSConv(in_channels, kernel_size, groups, split, norm, act)
    else:
        return Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      groups=groups, norm=norm, act=act)