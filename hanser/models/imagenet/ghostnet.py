import math
import tensorflow as tf
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d


class GhostConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, ratio=2, dw_size=3,
                 norm='def', act='def'):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio-1)

        self.primary_conv = Conv2d(
            in_channels, init_channels, kernel_size, norm=norm, act=act)

        self.cheap_operation = Conv2d(
            init_channels, new_channels, dw_size, groups=init_channels, norm=norm, act=act)

    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1, x2], axis=-1)
        if out.shape[-1] != self.out_channels:
            out = out[:, :, :, :self.out_channels]
        return out