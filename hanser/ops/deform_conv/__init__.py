# import tensorflow as tf
#
# def deform_conv(input, weight, offset, kernel_size, stride=1, padding=0, dilation=1):
#     # input: (N, iH, iW, iC)
#     # weight: (kH, kW, iC, oC)
#     # offset: (N, oH, oW, 2*kH*kW)
#     # Returns:
#     #   output: (N, oH, oW, oC)
#     # im2col: (N, H, W, iC) -> (N, oH, oW, iC, kH*kW)
#     # (N, oH, oW, iC, kH*kW) @ (kH*kW, iC, oC) -> (N, oH, oW, oC)
#
#     return


import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class DeformableConv2D(Conv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (in_channels, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * in_channels, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (in_channels, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        dtype = inputs.dtype
        # get offset, shape [N, oH, oW, kH * kW * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        shape = tf.shape(inputs)
        # some length
        N = shape[0]
        iH, iW, iC = inputs.shape[1:4]
        shape = tf.shape(offset)
        oC = self.filters
        oH, oW = offset.shape[1:3]
        kH, kW = self.kernel_size

        offset = tf.reshape(offset, [N, oH, oW, kH*kW, 2])
        offset_y, offset_x = offset[..., 0], offset[..., 1]

        y, x = self._get_conv_indices([iH, iW]) # [1, oH, oW, kH*kW]
        y, x = [tf.tile(t, [N, 1, 1, 1]) for t in [y, x]] # [N, oH, oW, kH*kW]
        y, x = [tf.cast(t, dtype) for t in [y, x]]

        y, x = y + offset_y, x + offset_x
        y = tf.clip_by_value(y, 0, iH - 1.0)
        x = tf.clip_by_value(x, 0, iW - 1.0)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(t), dtype=tf.int32) for t in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, x0 = [tf.maximum(t, 0) for t in [y0, x0]]
        y1, x1 = tf.minimum(x1, iH - 1), tf.minimum(x1, iW - 1)

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, t) for t in indices]
        # (N, oH, oW, kH*kW, iC)

        # cast to float
        x0, x1, y0, y1 = [tf.cast(t, dtype) for t in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(t, axis=-1) for t in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [N, oH, oW, kH, kW, iC])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5])
        pixels = tf.reshape(pixels, [N, oH*kW, oW*kW, iC, 1])

        pixels = tf.tile(pixels, [1, 1, 1, 1, oC])
        pixels = tf.reshape(pixels, [N, oH*kH, oW*kW, oC*iC])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, kH, kW, 1], 'VALID')
        out = tf.reshape(out, [N, oH, oW, oC, iC])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size
        :return: y, x with shape [1, oH, oW, kH*kW]
        """
        iH, iW = feature_map_size

        y, x = tf.meshgrid(tf.range(iH), tf.range(iW))
        y, x = [t[None, :, :, None] for t in [y, x]]  # [1, iH, iW, 1]
        y, x = [tf.image.extract_patches(
            t, [1, *self.kernel_size, 1], [1, *self.strides, 1],
            [1, *self.dilation_rate, 1], 'VALID') for t in [y, x]]  # [1, oH, oW, kH*kW]
        return y, x

    def _get_pixel_values_at_point(self, inputs, indices):
        """get pixel values
        Args:
            inputs: (N, iH, iW, iC)
            indices: (N, oH, oW, kH*kW)
        Returns:
            (N, oH, oW, kH*kW, iC)
        """
        y, x = indices
        shape = tf.shape(y)
        N, H, W, I = shape[0], shape[1], shape[2], shape[3]

        batch_idx = tf.reshape(tf.range(N), (N, 1, 1, 1))
        b = tf.tile(batch_idx, (1, H, W, I))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)