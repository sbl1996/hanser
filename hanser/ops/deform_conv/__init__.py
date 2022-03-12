import tensorflow as tf


def _pad_input(input, kernel_size, stride, padding, dilation):
    if padding == 'same':
        hw = input.shape[1:3]
        padding_list = []
        for i in range(2):
            k = kernel_size[i]
            s = stride[i]
            d = dilation[i]
            dk = k + (k - 1) * (d - 1)
            same_output = (hw[i] + s - 1) // s
            valid_output = (hw[i] - dk + s) // s
            if same_output == valid_output:
                padding_list += [0, 0]
            else:
                p = dk - 1
                p_0 = p // 2
                padding_list += [p_0, p - p_0]
        if sum(padding_list) != 0:
            padding = [[0, 0],
                       [padding_list[0], padding_list[1]],  # top, bottom padding
                       [padding_list[2], padding_list[3]],  # left, right padding
                       [0, 0]]
            input = tf.pad(input, padding)
    return input


def _get_pixel_values_at_point(inputs, indices):
    """get pixel values
    Args:
        inputs: (N, iH, iW, iC)
        indices: (N, oH, oW, kH*kW)
    Returns:
        (N, oH, oW, kH*kW, iC)
    """
    y, x = [tf.cast(t, tf.int32) for t in indices]
    shape = tf.shape(y)
    N, H, W, I = shape[0], shape[1], shape[2], shape[3]

    batch_idx = tf.reshape(tf.range(N), (N, 1, 1, 1))
    b = tf.tile(batch_idx, (1, H, W, I))
    pixel_idx = tf.stack([b, y, x], axis=-1)
    return tf.gather_nd(inputs, pixel_idx)


def _get_conv_indices(size, kernel_size, stride, dilation):
    """the x, y coordinates in the window when a filter sliding on the feature map
    Returns:
        # (1, oH, oW, kH*kW)
    """
    iH, iW = size
    y, x = tf.meshgrid(tf.range(iH), tf.range(iW))
    y, x = [t[None, :, :, None] for t in [y, x]]  # [1, iH, iW, 1]
    y, x = [tf.image.extract_patches(
        t, [1, *kernel_size, 1], [1, *stride, 1],
        [1, *dilation, 1], 'VALID') for t in [y, x]]  # [1, oH, oW, kH*kW]
    return y, x


def deform_conv(input, weight, offset, kernel_size, stride=1, padding=0, dilation=1):
    # input: (N, iH, iW, iC)
    # weight: (kH, kW, iC, oC)
    # offset: (N, oH, oW, kH*kW*2)
    # Returns:
    #   output: (N, oH, oW, oC)
    # im2col: (N, H, W, iC) -> (N, oH, oW, iC, kH*kW)
    # (N, oH, oW, iC, kH*kW) @ (kH*kW, iC, oC) -> (N, oH, oW, oC)
    dtype = offset.dtype

    input = _pad_input(input, kernel_size, stride, padding, dilation)

    shape = tf.shape(input)
    N = shape[0]
    iH, iW, iC = input.shape[1:4]
    oH, oW = offset.shape[1:3]
    kH, kW = kernel_size

    offset = tf.reshape(offset, [N, oH, oW, kH * kW, 2])
    offset_y, offset_x = offset[..., 0], offset[..., 1]

    y, x = _get_conv_indices([iH, iW], kernel_size, stride, dilation)  # [1, oH, oW, kH*kW]
    y, x = [tf.tile(t, [N, 1, 1, 1]) for t in [y, x]]  # [N, oH, oW, kH*kW]
    y, x = [tf.cast(t, dtype) for t in [y, x]]

    y, x = y + offset_y, x + offset_x
    y = tf.clip_by_value(y, 0, iH - 1.0)
    x = tf.clip_by_value(x, 0, iW - 1.0)

    y0, x0 = [tf.floor(t) for t in [y, x]]
    y1, x1 = y0 + 1, x0 + 1
    y0, x0 = [tf.maximum(t, 0) for t in [y0, x0]]
    y1, x1 = tf.minimum(x1, iH - 1), tf.minimum(x1, iW - 1)

    indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
    p0, p1, p2, p3 = [_get_pixel_values_at_point(input, t) for t in indices]
    # (N, oH, oW, kH*kW, iC)

    w0 = (y1 - y) * (x1 - x)
    w1 = (y1 - y) * (x - x0)
    w2 = (y - y0) * (x1 - x)
    w3 = (y - y0) * (x - x0)
    w0, w1, w2, w3 = [tf.expand_dims(t, axis=-1) for t in [w0, w1, w2, w3]]
    input = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

    input = tf.reshape(input, [N, oH, oW, kH * kW, iC])
    # input = tf.transpose(input, [0, 1, 2, 4, 3])
    weight = tf.reshape(weight, (kH*kW, *weight.shape[2:]))
    # out = tf.tensordot(input, self.kernel, axes=2)
    output = tf.einsum('nhwki,kio->nhwo', input, weight)
    return output