import math
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec, Dropout, Layer, Conv2D
from tensorflow.keras.initializers import Constant


class PadChannel(Layer):

    def __init__(self, c, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def call(self, x):
        return tf.pad(x, [(0, 0), (0, 0), (0, 0), (0, self.c)])

    def compute_output_shape(self, input_shape):
        in_channels = input_shape[-1]
        return input_shape[:-1] + (in_channels + self.c,)

    def get_config(self):
        return {
            **super().get_config(),
            'c': self.c,
        }


class StochDepth(Layer):

    def __init__(self, drop_rate, scale_by_keep=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = self.add_weight(
            name="drop_rate", shape=(), dtype=tf.float32,
            initializer=initializers.Constant(drop_rate), trainable=False)
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=None):
        if not training:
            return x
        noise_shape = (tf.shape(x)[0], 1, 1, 1)
        r = tf.random.uniform(noise_shape, dtype=x.dtype)
        keep_prob = 1. - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'drop_rate': self.drop_rate,
            'scale_by_keep': self.scale_by_keep,
        }


class DropPath(Dropout):

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = self.add_weight(
            name="drop_rate", shape=(), dtype=tf.float32,
            initializer=initializers.Constant(rate), trainable=False)

    def call(self, inputs, training=None):

        if training:
            noise_shape = (tf.shape(inputs)[0], 1, 1, 1)
            return tf.nn.dropout(
                inputs,
                noise_shape=noise_shape,
                rate=self.rate)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'rate': self.rate,
        }


class ReZero(Layer):

    def __init__(self, init_val=0., **kwargs):
        super().__init__(**kwargs)
        self.init_val = init_val
        self.res_weight = self.add_weight(
            name='res_weight', shape=(), dtype=tf.float32,
            trainable=True, initializer=initializers.Constant(init_val))

    def call(self, x):
        return x * self.res_weight

    def get_config(self):
        return {
            **super().get_config(),
            "init_val": self.init_val,
        }


class Affine(Layer):

    def __init__(self,
                 axis=-1,
                 scale=True,
                 center=True,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)
        self.scale = scale
        self.center = center
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)

    def call(self, inputs):
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1)):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        if scale is not None:
            inputs = inputs * scale

        if offset is not None:
            inputs = inputs + offset
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, inputs_dtype)
        return inputs

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                 input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
            ]

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                trainable=self.trainable,
                experimental_autocast=False)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                trainable=self.trainable,
                experimental_autocast=False)
        else:
            self.beta = None

    def get_config(self):
        base_config = super(Affine, self).get_config()
        base_config = {
            "scale": self.scale,
            "center": self.center,
            **base_config
        }
        return base_config


class AntiAliasing(Layer):

    def __init__(self, kernel_size=3, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, input_shape):
        kernel_size = self.kernel_size
        padding = int((kernel_size - 1) // 2)
        self.paddings = [
            [0, 0], [padding, padding], [padding, padding], [0, 0]
        ]

        if kernel_size == 1:
            a = np.array([1., ])
        elif kernel_size == 2:
            a = np.array([1., 1.])
        elif kernel_size == 3:
            a = np.array([1., 2., 1.])
        elif kernel_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif kernel_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif kernel_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif kernel_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError("Not supported kernel_size: %d" % kernel_size)

        G = input_shape[-1]

        kernel = a[:, None] * a[None, :]
        kernel = kernel / kernel.sum()
        kernel = np.tile(kernel[:, :, None, None], [1, 1, G, 1])

        self.kernel = self.add_weight(
            name="kernel", shape=kernel.shape, dtype=self.dtype,
            initializer=initializers.Constant(kernel), trainable=False)

    def call(self, inputs, training=None):
        stride = self.stride

        if self.kernel_size == 1:
            return inputs[:, :, ::stride, ::stride]
        else:
            # inputs = tf.pad(inputs, self.paddings, "REFLECT")
            inputs = tf.pad(inputs, self.paddings)
            strides = (1, stride, stride, 1)
            output = tf.nn.depthwise_conv2d(
                inputs, self.kernel, strides=strides, padding='VALID')
            return output

    def get_config(self):
        base_config = super().get_config()
        base_config = {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            **base_config
        }
        return base_config


class SpaceToDepth(Layer):

    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, self.block_size)

    def get_config(self):
        return {
            **super().get_config(),
            "block_size": self.block_size,
        }


class Slice(Layer):

    def __init__(self, begin, size, **kwargs):
        super().__init__(**kwargs)
        self.begin = begin
        self.size = size
        assert len(begin) == len(size)

    def compute_output_shape(self, input_shape):
        begins = self.begin[1:]
        sizes = self.size[1:]
        ends = [shape if size == -1 else begin + size for (begin, size, shape) in zip(
            begins, sizes, input_shape[1:])]
        sizes = [end - begin for begin, end in zip(begins, ends)]
        output_shape = (input_shape[0], *sizes)
        return output_shape

    def call(self, x):
        return tf.slice(x, (0, *self.begin), (-1, *self.size))


class DropBlock(Layer):

    def __init__(self, keep_prob, block_size, gamma_scale=1., per_channel=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        self.block_size = block_size
        self.gamma_scale = gamma_scale
        self.per_channel = per_channel

        self.keep_prob = self.add_weight(
            name="keep_prob", shape=(), dtype=tf.float32,
            initializer=Constant(keep_prob), trainable=False,
            experimental_autocast=False,
        )

    def call(self, x, training=None):
        if training:
            n = tf.shape(x)[0]
            h, w, c = x.shape[1:]
            by, bx = self.block_size
            # by = min(h, by)
            # bx = min(w, bx)

            t = (by - 1) // 2
            l = (bx - 1) // 2
            b = (by - 1) - t
            r = (bx - 1) - l

            c = c if self.per_channel else 1
            sampling_mask_shape = [n, h - by + 1, w - bx + 1, c]
            pad_shape = [[0, 0], [t, b], [l, r], [0, 0]]

            ratio = (w * h) / (bx * by) / ((w - bx + 1) * (h - by + 1))
            gamma = (1. - self.keep_prob) * ratio * self.gamma_scale
            mask = tf.cast(
                tf.random.uniform(sampling_mask_shape) < gamma, tf.float32)
            mask = tf.pad(mask, pad_shape)

            mask = tf.nn.max_pool2d(mask, (by, bx), strides=1, padding='SAME')
            mask = 1. - mask

            axis = [1, 2] if self.per_channel else [1, 2, 3]
            mask_reduce_sum = tf.reduce_sum(mask, axis=axis, keepdims=True)
            normalize_factor = tf.cast(h * w, dtype=tf.float32) / (mask_reduce_sum + 1e-8)

            x = x * tf.cast(mask, x.dtype) * tf.cast(normalize_factor, x.dtype)
            return x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'keep_prob': self.keep_prob,
            "block_size": self.block_size,
            "gamma_scale": self.gamma_scale,
            "per_channel": self.per_channel,
        }


class ScaledWSConv2D(Conv2D):

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
            experimental_autocast=False)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__

        default_conv_op = partial(
            tf.nn.convolution,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)

        self.gain = self.add_weight(
            "gain", shape=(self.filters,), dtype=tf.float32,
            trainable=True, initializer='ones', experimental_autocast=False)

        def standardized_conv_op(inputs, kernel):
            fan_in = tf.constant(np.prod(kernel.shape[:3]), kernel.dtype)
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            kernel = (kernel - mean) / tf.sqrt(var * fan_in + 1e-4)
            kernel = kernel * self.gain
            kernel = tf.cast(kernel, inputs.dtype)
            return default_conv_op(inputs, kernel)

        self._convolution_op = standardized_conv_op
        self.built = True