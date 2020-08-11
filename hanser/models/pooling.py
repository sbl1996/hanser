from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn


class Pooling2D(Layer):
    """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).

    This class only exists for code reuse. It will never be an exposed API.

    Arguments:
      pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(self, pool_function, pool_size, strides,
                 padding='valid', data_format=None,
                 name=None, **kwargs):
        super(Pooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self._horch_impl = False
        if self.pool_size[0] == 3 and self.padding == 'same':
            assert self.pool_size[0] == self.pool_size[1]
            assert self.data_format == 'channels_last'
            self._horch_impl = True

    def call(self, inputs):
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        if self._horch_impl:
            outputs = self.pool_function(
                inputs[:, ::-1, ::-1, :],
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format, 4))[:, ::-1, ::-1, :]
        else:
            outputs = self.pool_function(
                inputs,
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format, 4))
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Pooling2D):
    """Max pooling operation for 2D spatial data.

    Downsamples the input representation by taking the maximum value over the
    window defined by `pool_size` for each dimension along the features axis.
    The window is shifted by `strides` in each dimension.  The resulting output
    when using "valid" padding option has a shape(number of rows or columns) of:
    `output_shape = (input_shape - pool_size + 1) / strides)`

    The resulting output shape when using the "same" padding option is:
    `output_shape = input_shape / strides`

    For example, for stride=(1,1) and padding="valid":

    >>> x = tf.constant([[1., 2., 3.],
    ...                  [4., 5., 6.],
    ...                  [7., 8., 9.]])
    >>> x = tf.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding='valid')
    >>> max_pool_2d(x)
    <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
      array([[[[5.],
               [6.]],
              [[8.],
               [9.]]]], dtype=float32)>

    For example, for stride=(2,2) and padding="valid":

    >>> x = tf.constant([[1., 2., 3., 4.],
    ...                  [5., 6., 7., 8.],
    ...                  [9., 10., 11., 12.]])
    >>> x = tf.reshape(x, [1, 3, 4, 1])
    >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding='valid')
    >>> max_pool_2d(x)
    <tf.Tensor: shape=(1, 2, 3, 1), dtype=float32, numpy=
      array([[[[ 6.],
               [ 7.],
               [ 8.]],
              [[10.],
               [11.],
               [12.]]]], dtype=float32)>

    Usage Example:

    >>> input_image = tf.constant([[[[1.], [1.], [2.], [4.]],
    ...                            [[2.], [2.], [3.], [2.]],
    ...                            [[4.], [1.], [1.], [1.]],
    ...                            [[2.], [2.], [1.], [4.]]]])
    >>> output = tf.constant([[[[1], [0]],
    ...                       [[0], [1]]]])
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    input_shape=(4,4,1)))
    >>> model.compile('adam', 'mean_squared_error')
    >>> model.predict(input_image, steps=1)
    array([[[[2.],
             [4.]],
            [[4.],
             [4.]]]], dtype=float32)

    For example, for stride=(1,1) and padding="same":

    >>> x = tf.constant([[1., 2., 3.],
    ...                  [4., 5., 6.],
    ...                  [7., 8., 9.]])
    >>> x = tf.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding='same')
    >>> max_pool_2d(x)
    <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
      array([[[[5.],
               [6.],
               [6.]],
              [[8.],
               [9.],
               [9.]],
              [[8.],
               [9.],
               [9.]]]], dtype=float32)>

    Arguments:
      pool_size: integer or tuple of 2 integers,
        window size over which to take the maximum.
        `(2, 2)` will take the max value over a 2x2 pooling window.
        If only one integer is specified, the same window length
        will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
        Strides values.  Specifies how far the pooling window moves
        for each pooling step. If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        "valid" adds no zero padding.  "same" adds padding such that if the stride
        is 1, the output shape is the same as input shape.
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
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

    Returns:
      A tensor of rank 4 representing the maximum pooled values.  See above for
      output shape.
    """

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 data_format=None,
                 **kwargs):
        super(MaxPooling2D, self).__init__(
            nn.max_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, **kwargs)


class AveragePooling2D(Pooling2D):
    """Average pooling operation for spatial data.

    Arguments:
      pool_size: integer or tuple of 2 integers,
        factors by which to downscale (vertical, horizontal).
        `(2, 2)` will halve the input in both spatial dimension.
        If only one integer is specified, the same window length
        will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
        Strides values.
        If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
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
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
    """

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 data_format=None,
                 **kwargs):
        super(AveragePooling2D, self).__init__(
            nn.avg_pool,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, **kwargs)
