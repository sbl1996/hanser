import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn


class Pooling2D(Layer):

    def __init__(self, pool_function, pool_size, strides, padding='valid', **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self._horch_impl = False
        if self.pool_size[0] == 3 and self.padding == 'same':
            assert self.pool_size[0] == self.pool_size[1]
            self._horch_impl = True

    def call(self, inputs):
        pool_shape = (1,) + self.pool_size + (1,)
        strides = (1,) + self.strides + (1,)
        if self._horch_impl:
            inputs = tf.reverse(inputs, [1, 2])
            outputs = self.pool_function(
                inputs,
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper())
            outputs = tf.reverse(outputs, [1, 2])
        else:
            outputs = self.pool_function(
                inputs,
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper())
        return outputs

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Pooling2D):

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 **kwargs):
        super().__init__(
            nn.max_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, **kwargs)


class AveragePooling2D(Pooling2D):

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 **kwargs):
        super().__init__(
            nn.avg_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, **kwargs)
