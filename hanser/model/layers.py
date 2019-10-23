import tensorflow as tf

from tensorflow.python.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Conv2DTranspose, DepthwiseConv2D, \
    GlobalAvgPool2D, Flatten, ReLU, Activation, Multiply

from hanser.model import get_default


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


def se(x, channels, ratio):
    s = GlobalAvgPool2D()(x)
    s = Flatten()(s)
    s = dense(s, int(channels * ratio))
    s = ReLU()(s)
    s = dense(s, channels)
    s = sigmoid(s)
    x = Multiply()([x, s])
    return x


def sigmoid(x):
    return Activation('sigmoid')(x)


def swish(x):
    return Multiply()([x, sigmoid(x)])


def conv2d(x, channels, kernel_size, stride=1, padding='same', use_bias=False):
    l2 = get_default('l2_regularizer')
    kernel_regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    return Conv2D(channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                  kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(x)


def dwconv2d(x, kernel_size, stride=1, padding='same', use_bias=False):
    l2 = get_default('l2_regularizer')
    kernel_regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    return DepthwiseConv2D(kernel_size, stride, padding, use_bias=use_bias,
                           depthwise_initializer='he_normal', depthwise_regularizer=kernel_regularizer)(x)


def deconv2d(x, channels, kernel_size, stride=1, padding='same', use_bias=False):
    l2 = get_default('l2_regularizer')
    kernel_regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    bias_regularizer = tf.keras.regularizers.l2(l2) if (l2 and use_bias) else None
    return Conv2DTranspose(channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                           kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer)(x)


def bn(x, fused=None, gamma='ones', training=True):
    if fused is None:
        fused = get_default(['bn', 'fused'])
    momentum = get_default(['bn', 'momentum'])
    epsilon = get_default(['bn', 'epsilon'])
    return BatchNormalization(fused=fused, gamma_initializer=gamma, momentum=momentum, epsilon=epsilon)(x, training=training)


def dense(x, channels):
    l2 = get_default('l2_regularizer')
    kernel_regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    bias_regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    return Dense(channels, kernel_initializer='he_normal',
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)


def drop_connect(x, drop_rate):
    keep_prob = 1.0 - drop_rate

    batch_size = tf.shape(x)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    x = tf.div(x, keep_prob) * binary_tensor
    return x


class DropConnect(Layer):

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
