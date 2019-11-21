import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Conv2DTranspose, DepthwiseConv2D, \
    GlobalAvgPool2D, Flatten, ReLU, Activation, Multiply, Concatenate, Add

from hanser.tpu import TPUBatchNormalization
from hanser.model import get_default


class SE(Model):

    def __init__(self, channels, ratio, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.ratio = ratio
        self.fc = Sequential([
            conv2d(int(channels * ratio), 1),
            ReLU(),
            conv2d(channels, 1),
            Sigmoid(),
        ])

    def call(self, x):
        s = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        s = self.fc(s)
        return x * s


class Sigmoid(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        return tf.nn.sigmoid(x)


class Swish(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        return x * tf.nn.sigmoid(x)


def conv2d(channels, kernel_size, stride=1, padding='same', dilation=1, use_bias=False, kernel_initializer='he_normal',
           bias_initializer='zeros', name=None):
    return Conv2D(channels, kernel_size=kernel_size, strides=stride,
                  padding=padding, dilation_rate=dilation, use_bias=use_bias,
                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                  name=name)


def dwconv2d(kernel_size, stride=1, padding='same', depth_multiplier=1, use_bias=False, dilation=1, name=None):
    return DepthwiseConv2D(kernel_size, stride, padding, use_bias=use_bias,
                           dilation_rate=dilation,
                           depth_multiplier=depth_multiplier,
                           depthwise_initializer='he_normal',
                           name=name)


def deconv2d(channels, kernel_size, stride=1, padding='same', use_bias=False):
    return Conv2DTranspose(channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                           kernel_initializer='he_normal')


def bn(fused=None, gamma='ones', affine=True, name=None):
    if fused is None:
        fused = get_default(['bn', 'fused'])
    momentum = get_default(['bn', 'momentum'])
    epsilon = get_default(['bn', 'epsilon'])
    center = scale = affine
    if get_default(['bn', 'tpu']):
        return TPUBatchNormalization(fused=False, gamma_initializer=gamma, momentum=momentum, epsilon=epsilon,
                                     center=center, scale=scale, name=name)
    else:
        return BatchNormalization(fused=fused, gamma_initializer=gamma, momentum=momentum, epsilon=epsilon,
                                  center=center, scale=scale, name=name)


def dense(channels, name=None):
    return Dense(channels, kernel_initializer='he_normal', name=name)
