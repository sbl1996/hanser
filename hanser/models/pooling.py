import tensorflow as tf
from tensorflow.keras.layers import Layer


def normalize_padding(value):
  if isinstance(value, (list, tuple)):
    return value
  padding = value.lower()
  if padding not in {'valid', 'same', 'causal'}:
    raise ValueError('The `padding` argument must be a list/tuple or one of '
                     '"valid", "same" (or "causal", only for `Conv1D). '
                     'Received: ' + str(padding))
  return padding


def normalize_tuple(value, n, name):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Arguments:
    value: The value to validate and convert. Could an int, or any iterable of
      ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    if len(value_tuple) != n:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(n) + ' integers. Received: ' + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError('The `' + name + '` argument must be a tuple of ' +
                         str(n) + ' integers. Received: ' + str(value) + ' '
                         'including element ' + str(single_value) + ' of type' +
                         ' ' + str(type(single_value)))
    return value_tuple


class Pooling2D(Layer):

    def __init__(self, pool_function, pool_size, strides, padding='valid', **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = normalize_tuple(strides, 2, 'strides')
        self.padding = normalize_padding(padding)
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
            tf.nn.max_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, **kwargs)


class AveragePooling2D(Pooling2D):

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 **kwargs):
        super().__init__(
            tf.nn.avg_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, **kwargs)
