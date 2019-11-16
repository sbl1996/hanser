import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Conv2DTranspose, DepthwiseConv2D, \
    GlobalAvgPool2D, Flatten, ReLU, Activation, Multiply
from tensorflow.python.keras.utils import conv_utils

from hanser.tpu import TPUBatchNormalization
from hanser.model import get_default
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec


class PadChannel(Layer):

    def __init__(self, out_channels, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels

    def call(self, x):
        c = tf.subtract(self.out_channels, tf.shape(x)[-1])
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


def conv2d(x, channels, kernel_size, stride=1, padding='same', dilation=1, use_bias=False, kernel_initializer='he_normal', bias_initializer='zeros', name=None):
    return Conv2D(channels, kernel_size=kernel_size, strides=stride,
                  padding=padding, dilation_rate=dilation, use_bias=use_bias,
                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                  name=name)(x)


def dwconv2d(x, kernel_size, stride=1, padding='same', depth_multiplier=1, use_bias=False, name=None):
    return DepthwiseConv2D(kernel_size, stride, padding, use_bias=use_bias,
                           depth_multiplier=depth_multiplier,
                           depthwise_initializer='he_normal',
                           name=name)(x)


def deconv2d(x, channels, kernel_size, stride=1, padding='same', use_bias=False):
    return Conv2DTranspose(channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                           kernel_initializer='he_normal')(x)


def relu(x, name=None):
    return ReLU(name=name)(x)


def bn(x, fused=None, gamma='ones', name=None):
    if fused is None:
        fused = get_default(['bn', 'fused'])
    momentum = get_default(['bn', 'momentum'])
    epsilon = get_default(['bn', 'epsilon'])
    if get_default(['bn', 'tpu']):
        return TPUBatchNormalization(fused=False, gamma_initializer=gamma, momentum=momentum, epsilon=epsilon,
                                     name=name)(x)
    else:
        return BatchNormalization(fused=fused, gamma_initializer=gamma, momentum=momentum, epsilon=epsilon, name=name)(x)


def dense(x, channels, name=None):
    return Dense(channels, kernel_initializer='he_normal', name=name)(x)


def drop_connect(x, drop_rate):
    keep_prob = 1.0 - drop_rate

    batch_size = tf.shape(x)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=x.dtype)
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


class ReflectionPad2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

  This layer can add rows and columns of zeros
  at the top, bottom, left and right side of an image tensor.

  Arguments:
    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch, padded_rows, padded_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch, channels, padded_rows, padded_cols)`
  """

    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs):
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], list(self.padding[0]), list(self.padding[1])]
        else:
            pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]
        return tf.pad(inputs, pattern, mode='REFLECT')

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelShuffle(Layer):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    groups : int
        Number of groups.
    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    def __init__(self, groups, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.groups = groups

    def call(self, x):
        if self.data_format == 'channels_first':
            batch, channels, height, width = tf.shape(x)
        else:
            batch, height, width, channels = tf.shape(x)

        # assert (channels % groups == 0)
        channels_per_group = channels // self.groups

        if self.data_format == 'channels_first':
            x = K.reshape(x, shape=(-1, self.groups, channels_per_group, height, width))
            x = K.permute_dimensions(x, pattern=(0, 2, 1, 3, 4))
            x = K.reshape(x, shape=(-1, channels, height, width))
        else:
            x = K.reshape(x, shape=(-1, height, width, self.groups, channels_per_group))
            x = K.permute_dimensions(x, pattern=(0, 1, 2, 4, 3))
            x = K.reshape(x, shape=(-1, height, width, channels))
        return x

    def get_config(self):
        config = {'groups': self.groups, 'data_format': self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_custom_objects():
    return {
        'ReflectionPad2D': ReflectionPad2D,
        'ChannelShuffle': ChannelShuffle,
    }