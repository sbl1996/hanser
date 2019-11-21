import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import MaxPool2D, AvgPool2D, Layer, ReLU

from hanser.model.layers2 import bn, conv2d, dwconv2d


class FactorizedReduce(Model):

    def __init__(self, channels, name=None):
        super().__init__(name=name)
        assert channels % 2 == 0
        self.channels = channels
        self.conv1 = conv2d(channels // 2, 1, 2)
        self.conv2 = conv2d(channels // 2, 1, 2)
        self.bn = bn()

    def call(self, x):
        x = tf.nn.relu(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, 1:, 1:, :])
        x = tf.concat([x1, x2], axis=-1)
        x = self.bn(x)
        return x


class Zero(Layer):

    def __init__(self, stride, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride

    def call(self, x, training=None):
        if self.stride == 2:
            x = x[:, ::2, ::2, :]
        return x * 0

    def get_config(self):
        config = super().get_config()
        config.update({'stride': self.stride})
        return config


class Identity(Layer):

    def call(self, x, training=None):
        return x


def sep_conv(channels, kernel_size, stride):
    return Sequential([
        ReLU(),
        dwconv2d(kernel_size, stride=stride),
        conv2d(channels, 1),
        bn(affine=False),
        ReLU(),
        dwconv2d(kernel_size),
        conv2d(channels, 1),
        bn(affine=False)
    ])


def dil_conv(channels, kernel_size, dilation):
    return Sequential([
        ReLU(),
        dwconv2d(kernel_size, dilation=dilation),
        conv2d(channels, 1),
        bn(affine=False)
    ])


def sep_conv_3x3(channels, stride):
    return sep_conv(channels, 3, stride)


def sep_conv_5x5(channels, stride):
    return sep_conv(channels, 3, stride)


def skip_connect(channels, stride):
    return Identity() if stride == 1 else FactorizedReduce(channels)


def dil_conv_3x3(channels):
    return dil_conv(channels, 3, 2)


def avg_pool_3x3(stride):
    return AvgPool2D(3, stride, padding='same')


def max_pool_3x3(stride):
    return MaxPool2D(3, stride, padding='same')


NORMAL_OPS = {
    'none': lambda c, s: Zero(1),
    'skip_connect': lambda c, s: skip_connect(c, 1),
    'sep_conv_3x3': lambda c, s: sep_conv(c, 3, 1),
    'sep_conv_5x5': lambda c, s: sep_conv(c, 5, 1),
    'sep_conv_7x7': lambda c, s: sep_conv(c, 7, 1),
    'avg_pool_3x3': lambda c, s: avg_pool_3x3(1),
    'max_pool_3x3': lambda c, s: max_pool_3x3(1),
}

REDUCTION_OPS = {
    'none': lambda c, s: Zero(s),
    'skip_connect': lambda c, s: skip_connect(c, s),
    'avg_pool_3x3': lambda c, s: avg_pool_3x3(s),
    'max_pool_3x3': lambda c, s: max_pool_3x3(s),
}