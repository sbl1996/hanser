import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, GlobalAvgPool2D

from hanser.models.layers import Linear


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
        return dict(list(base_config.items()) + list(config.items()))


class SELayer(Layer):

    def __init__(self, in_channels, reduction):
        super().__init__()
        channels = in_channels // reduction
        self.pool = GlobalAvgPool2D()
        self.fc = Sequential(
            Linear(in_channels, channels, act='relu'),
            Linear(channels, in_channels, act='sigmoid'),
        )

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


def drop_connect(x, drop_rate):
    keep_prob = 1.0 - drop_rate

    batch_size = tf.shape(x)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    x = tf.div(x, keep_prob) * binary_tensor
    return x


class DropPath(Layer):

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_inputs():
            return drop_connect(x, self.rate)

        output = tf.cond(training, dropped_inputs, lambda: x)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
