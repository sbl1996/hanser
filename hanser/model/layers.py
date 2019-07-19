import tensorflow as tf

from tensorflow.python.keras.layers import Layer, Conv2D, BatchNormalization, Dense


class PadChannel(Layer):

    def __init__(self, out_channels, **kwargs):
        self.out_channels = out_channels
        super().__init__(**kwargs)

    def call(self, x):
        c = self.out_channels - x.shape[-1].value
        return tf.pad(x, [(0, 0), (0, 0), (0, 0), (0, c)])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_channels,)

    def get_config(self):
        config = {'out_channels': self.out_channels}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv2d(channels, kernel_size, stride=1, padding='same', use_bias=False):
    return Conv2D(channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias, kernel_initializer='he_normal')


def bn(fused=True, gamma='ones'):
    return BatchNormalization(fused=fused, gamma_initializer=gamma)

def dense(channels):
    return Dense(channels, kernel_initializer='he_normal')