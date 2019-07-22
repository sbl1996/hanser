import math

import tensorflow as tf

_MAX_LEVEL = 10
_FILL_COLOR = (128, 128, 128)


def random_crop(x, size, padding, fill=128):
    height, width = size
    ph, pw = padding
    x = tf.pad(x, [(ph, ph), (pw, pw), (0, 0)], constant_values=fill)
    x = tf.image.random_crop(x, [height, width, x.shape[-1]])
    return x


def invert(image):
    return 255 - image


def cutout(image, length):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    cy = tf.random_uniform((), 0, h, dtype=tf.int32)
    cx = tf.random_uniform((), 0, w, dtype=tf.int32)

    t = tf.maximum(0, cy - length // 2)
    b = tf.minimum(h, cy + length // 2)
    l = tf.maximum(0, cx - length // 2)
    r = tf.minimum(w, cx + length // 2)
    shape = [b - t, r - l]
    padding = [(t, h - b), (l, w - r)]

    mask = tf.pad(tf.zeros(shape, dtype=image.dtype), padding, constant_values=1)

    mask = tf.expand_dims(mask, -1)
    image = image * mask
    return image


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
        image1: An image Tensor of type uint8.
        image2: An image Tensor of type uint8.
        factor: A floating point value above 0.0.
    Returns:
        A blended image Tensor of type uint8.
  """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled
    # Interpolate

    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)


def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
        image: An image Tensor of type uint8.
        degrees: Float, a scalar angle in degrees to rotate all images by. If
            degrees is positive the image will be rotated clockwise otherwise it will
            be rotated counterclockwise.
        replace: A one or three value 1D tensor to fill empty pixels caused by
            the rotate operation.
    Returns:
        The rotated version of image.
    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = tf.contrib.image.rotate(wrap(image), radians)
    return unwrap(image, replace)


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)
    return extended


def unwrap(image, replace):
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.
    Args:
        image: A 3D Image Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3]

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.

    Args:
        image: A 3D uint8 tensor.
    Returns:
        The image after it has had autocontrast applied to it and will be of type
        uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant(
        [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
        shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding='VALID', rate=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def shear_x(image, magnitude, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = tf.contrib.image.transform(
        wrap(image), [1., magnitude, 0., 0., 1., 0., 0., 0.])
    return unwrap(image, replace)


def shear_y(image, magnitude, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = tf.contrib.image.transform(
        wrap(image), [1., 0., 0., magnitude, 1., 0., 0., 0.])
    return unwrap(image, replace)


def translate_x(image, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    image = tf.contrib.image.translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    image = tf.contrib.image.translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random_uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level


def _enhance_level_to_arg(level):
    # TODO: To complex control flow, may be fixed in TF 2.0
    # level = (level / _MAX_LEVEL) * 0.9
    # level = _randomly_negate_tensor(level)
    # return level
    return (level / _MAX_LEVEL) * 1.8 + 0.1


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * (32 * 150 / 331)
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level):
    level = (8, 8, 7, 7, 6, 6, 5, 5, 4, 4)[level]
    return level


def _solarize_level_to_arg(level):
    level = (_MAX_LEVEL - level + 1) / _MAX_LEVEL * 256
    return level


NAME_TO_FUNC = {
    "shearX": lambda img, level: shear_x(
        img, _shear_level_to_arg(level), _FILL_COLOR),
    "shearY": lambda img, level: shear_y(
        img, _shear_level_to_arg(level), _FILL_COLOR),
    'translateX': lambda img, level: translate_x(
        img, _translate_level_to_arg(level), _FILL_COLOR),
    'translateY': lambda img, level: translate_y(
        img, _translate_level_to_arg(level), _FILL_COLOR),
    "rotate": lambda img, level: rotate(
        img, _rotate_level_to_arg(level), _FILL_COLOR),
    "color": lambda img, level: color(
        img, _enhance_level_to_arg(level)),
    "posterize": lambda img, level: posterize(img, level),
    "solarize": lambda img, level: solarize(img, level),
    "contrast": lambda img, level: contrast(
        img, _enhance_level_to_arg(level)),
    "sharpness": lambda img, level: sharpness(
        img, _enhance_level_to_arg(level)),
    "brightness": lambda img, level: brightness(
        img, _enhance_level_to_arg(level)),
    "autocontrast": lambda img, level: autocontrast(img),
    "equalize": lambda img, level: equalize(img),
    "invert": lambda img, level: invert(img),
}


def _apply_func_with_prob(func, p, image, level):
    should_apply_op = tf.cast(
        tf.floor(tf.random_uniform([], dtype=tf.float32) + p), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, level),
        lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""

    policy_to_select = tf.random_uniform((), maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(policies):
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda: policy(image),
            lambda: image)
    return image


def sub_policy(p1, op1, level1, p2, op2, level2):
    def _apply_policy(image):
        image = _apply_func_with_prob(NAME_TO_FUNC[op1], p1, image, level1)
        image = _apply_func_with_prob(NAME_TO_FUNC[op2], p2, image, level2)
        return image
    return _apply_policy


def cifar10_policy():
    policies = [
        sub_policy(0.1, "invert", 7, 0.2, "contrast", 6),
        sub_policy(0.7, "rotate", 2, 0.3, "translateX", 9),
        sub_policy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
        sub_policy(0.5, "shearY", 8, 0.7, "translateY", 9),
        sub_policy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

        sub_policy(0.2, "shearY", 7, 0.3, "posterize", 7),
        sub_policy(0.4, "color", 3, 0.6, "brightness", 7),
        sub_policy(0.3, "sharpness", 9, 0.7, "brightness", 9),
        sub_policy(0.6, "equalize", 5, 0.5, "equalize", 1),
        sub_policy(0.6, "contrast", 7, 0.6, "sharpness", 5),

        sub_policy(0.7, "color", 7, 0.5, "translateX", 8),
        sub_policy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
        sub_policy(0.4, "translateY", 3, 0.2, "sharpness", 6),
        sub_policy(0.9, "brightness", 6, 0.2, "color", 8),
        sub_policy(0.5, "solarize", 2, 0.0, "invert", 3),

        sub_policy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
        sub_policy(0.2, "equalize", 8, 0.8, "equalize", 4),
        sub_policy(0.9, "color", 9, 0.6, "equalize", 6),
        sub_policy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
        sub_policy(0.1, "brightness", 3, 0.7, "color", 0),

        sub_policy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
        sub_policy(0.9, "translateY", 9, 0.7, "translateY", 9),
        sub_policy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
        sub_policy(0.8, "equalize", 8, 0.1, "invert", 3),
        sub_policy(0.7, "translateY", 9, 0.9, "autocontrast", 1),
    ]

    return policies


def autoaugment(image, augmentation_name):
    available_policies = {'CIFAR10': cifar10_policy}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))
    policies = available_policies[augmentation_name]()
    image = select_and_apply_random_policy(policies, image)
    return image
