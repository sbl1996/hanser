import math

from toolz import curry

import tensorflow as tf

from tensorflow_addons.image.translate_ops import translations_to_projective_transforms
from tensorflow_addons.image.transform_ops import angles_to_projective_transforms

from hanser.ops import log_uniform, prepend_dims
from hanser.transform.common import image_dimensions, to_4D_image, get_ndims, from_4D_image


IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.120, 57.375]


def transform(images, transforms, interpolation="BILINEAR", output_shape=None):
    image_or_images = tf.convert_to_tensor(images)
    transform_or_transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    if output_shape is None:
        output_shape = tf.shape(images)[1:3]

    output_shape = tf.convert_to_tensor(output_shape, tf.int32)

    if len(transform_or_transforms.get_shape()) == 1:
        transforms = transform_or_transforms[None]
    elif transform_or_transforms.get_shape().ndims is None:
        raise ValueError("transforms rank must be statically known")
    elif len(transform_or_transforms.get_shape()) == 2:
        transforms = transform_or_transforms
    else:
        transforms = transform_or_transforms
        raise ValueError(
            "transforms should have rank 1 or 2, but got rank %d"
            % len(transforms.get_shape())
        )

    output = tf.raw_ops.ImageProjectiveTransformV2(
        images=images,
        transforms=transforms,
        output_shape=output_shape,
        interpolation=interpolation.upper(),
    )
    return from_4D_image(output, original_ndims)


def _fill_region(shape, value, dtype):
    if value == 'normal':
        value = tf.random.normal(shape)
    elif value == 'uniform':
        value = tf.random.uniform(shape, 0, 256, dtype=tf.int32)
    else:
        value = tf.convert_to_tensor(value)
        if len(value.shape) == 0:
            value = tf.fill((shape[-1],), value)
        value = tf.tile(prepend_dims(value, len(shape) - 1), (*shape[:-1], 1))
    value = tf.cast(value, dtype)
    return value

@curry
def cutout3(image, length, fill=0):
    # Fastest version, recommended
    h, w, c = image_dimensions(image, 3)

    cy = tf.random.uniform((), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((), 0, w, dtype=tf.int32)

    length = length // 2
    t = tf.maximum(0, cy - length)
    b = tf.minimum(h, cy + length)
    l = tf.maximum(0, cx - length)
    r = tf.minimum(w, cx + length)

    top = image[:t, :, :]
    mid_left = image[t:b, :l, :]
    mid_right = image[t:b, r:, :]
    bottom = image[b:, :, :]

    fill = _fill_region((b - t, r - l, c), fill, image.dtype)

    mid = tf.concat([mid_left, fill, mid_right], 1)  # along x axis
    image = tf.concat([top, mid, bottom], 0)  # along y axis
    image.set_shape((h, w, c))
    return image


def invert(image):
    image = tf.convert_to_tensor(image)
    return 255 - image


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    uint8: [0, 255]
    float32: [0, 1]
    Args:
      image1: An image Tensor of type uint8 or float32.
      image2: An image Tensor of type uint8 or float32.
      factor: A floating point value above 0.0.
    Returns:
      A blended image Tensor of type uint8 or float32.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    assert image1.dtype == image2.dtype
    ori_dtype = image1.dtype

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    # noinspection PyChainedComparisons
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, ori_dtype)

    # Extrapolate:
    # We need to clip and then cast.
    if ori_dtype == tf.uint8:
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)
    else:
        return tf.clip_by_value(temp, 0.0, 1.0)


# def solarize(image, threshold=128):
#     # For each pixel in the image, select the pixel
#     # if the value is less than the threshold.
#     # Otherwise, subtract 255 from the pixel.
#     threshold = tf.cast(threshold, image.dtype)
#     return tf.where(image < threshold, image, 255 - image)


# def solarize_add(image, addition=0, threshold=128):
#     # For each pixel in the image less than threshold
#     # we add 'addition' amount to it and then clip the
#     # pixel value to be between 0 and 255. The value
#     # of 'addition' is between -128 and 128.
#     added_image = tf.cast(image, tf.int32) + addition
#     added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
#     return tf.where(image < threshold, added_image, image)


# uint8 or float32
def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


# uint8 or float32
def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    degenerate = tf.image.convert_image_dtype(degenerate, tf.uint8)

    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.math.bincount(degenerate, minlength=256, dtype=tf.int32)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    mean = tf.minimum(mean, 255.0) / 255.0
    degenerate = tf.fill(tf.shape(degenerate), mean)
    degenerate = tf.image.grayscale_to_rgb(degenerate)

    ori_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = blend(degenerate, image, factor)
    return tf.image.convert_image_dtype(image, ori_dtype)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


# def posterize(image, bits):
#     """Equivalent of PIL Posterize."""
#     bits = tf.cast(bits, image.dtype)
#     return tf.bitwise.left_shift(tf.bitwise.right_shift(image, bits), bits)
#
#
# def rotate(image, degrees, replace):
#     """Rotates the image by degrees either clockwise or counterclockwise.
#     Args:
#         image: An image Tensor of type uint8.
#         degrees: Float, a scalar angle in degrees to rotate all images by. If
#             degrees is positive the image will be rotated clockwise otherwise it will
#             be rotated counterclockwise.
#         replace: A one or three value 1D tensor to fill empty pixels caused by
#             the rotate operation.
#     Returns:
#         The rotated version of image.
#     """
#     # Convert from degrees to radians.
#     degrees_to_radians = math.pi / 180.0
#     radians = degrees * degrees_to_radians
#
#     # In practice, we should randomize the rotation degrees by flipping
#     # it negatively half the time, but that's done on 'degrees' outside
#     # of the function.
#     image_height = tf.cast(tf.shape(image)[0], tf.float32)
#     image_width = tf.cast(tf.shape(image)[1], tf.float32)
#     transforms = angles_to_projective_transforms(
#         radians, image_height, image_width)
#     image = transform(wrap(image), transforms)
#     return unwrap(image, replace)
#
#
# def wrap(image):
#     """Returns 'image' with an extra channel set to all 1s."""
#     shape = tf.shape(image)
#     extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
#     extended = tf.concat([image, extended_channel], 2)
#     return extended
#
#
# def unwrap(image, replace):
#     """Unwraps an image produced by wrap.
#     Where there is a 0 in the last channel for every spatial position,
#     the rest of the three channels in that spatial dimension are grayed
#     (set to 128).  Operations like translate and shear on a wrapped
#     Tensor will leave 0s in empty locations.  Some transformations look
#     at the intensity of values to do preprocessing, and we want these
#     empty pixels to assume the 'average' value, rather than pure black.
#     Args:
#         image: A 3D Image Tensor with 4 channels.
#         replace: A one or three value 1D tensor to fill empty pixels.
#     Returns:
#         image: A 3D image Tensor with 3 channels.
#     """
#     image_shape = tf.shape(image)
#     # Flatten the spatial dimensions.
#     flattened_image = tf.reshape(image, [-1, image_shape[2]])
#
#     # Find all pixels where the last channel is zero.
#     alpha_channel = flattened_image[:, 3][:, None]
#
#     replace = tf.constant(replace, tf.uint8)
#     if tf.rank(replace) == 0:
#         replace = tf.expand_dims(replace, 0)
#         replace = tf.concat([replace, replace, replace], 0)
#     replace = tf.concat([replace, tf.ones([1], dtype=image.dtype)], 0)
#
#     # Where they are zero, fill them in with 'replace'.
#     flattened_image = tf.where(
#         tf.equal(alpha_channel, 0),
#         tf.ones_like(flattened_image, dtype=image.dtype) * replace,
#         flattened_image)
#
#     image = tf.reshape(flattened_image, image_shape)
#     image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
#     return image
#
#
# def autocontrast(image):
#     """Implements Autocontrast function from PIL using TF ops.
#     Args:
#         image: A 3D uint8 tensor.
#     Returns:
#         The image after it has had autocontrast applied to it and will be of type
#         uint8.
#     """
#
#     def scale_channel(img):
#         """Scale the 2D image using the autocontrast rule."""
#         # A possibly cheaper version can be done using cumsum/unique_with_counts
#         # over the histogram values, rather than iterating over the entire image.
#         # to compute mins and maxes.
#         lo = tf.cast(tf.reduce_min(img), tf.float32)
#         hi = tf.cast(tf.reduce_max(img), tf.float32)
#
#         # Scale the image, making the lowest value 0 and the highest value 255.
#         def scale_values(im):
#             scale = 255.0 / (hi - lo)
#             offset = -lo * scale
#             im = tf.cast(im, tf.float32) * scale + offset
#             im = tf.clip_by_value(im, 0.0, 255.0)
#             return tf.cast(im, tf.uint8)
#
#         result = tf.cond(hi > lo, lambda: scale_values(img), lambda: img)
#         return result
#
#     # Assumes RGB for now.  Scales each channel independently
#     # and then stacks the result.
#     s1 = scale_channel(image[:, :, 0])
#     s2 = scale_channel(image[:, :, 1])
#     s3 = scale_channel(image[:, :, 2])
#     image = tf.stack([s1, s2, s3], 2)
#     return image
#
#
# def equalize(image):
#     """Implements Equalize function from PIL using TF ops."""
#
#     def scale_channel(im, c):
#         """Scale the data in the channel to implement equalize."""
#         im = tf.cast(im[:, :, c], tf.int32)
#         # Compute the histogram of the image channel.
#         histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
#
#         # For the purposes of computing the step, filter out the nonzeros.
#         nonzero = tf.where(tf.not_equal(histo, 0))
#         nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
#         step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255
#
#         # noinspection PyShadowingNames
#         def build_lut(histo, step):
#             # Compute the cumulative sum, shifting by step // 2
#             # and then normalization by step.
#             lut = (tf.cumsum(histo) + (step // 2)) // step
#             # Shift lut, prepending with 0.
#             lut = tf.concat([[0], lut[:-1]], 0)
#             # Clip the counts to be in range.  This is done
#             # in the C code for image.point.
#             return tf.clip_by_value(lut, 0, 255)
#
#         # If step is zero, return the original image.  Otherwise, build
#         # lut from the full histogram and step and then index from it.
#         result = tf.cond(tf.equal(step, 0),
#                          lambda: im,
#                          lambda: tf.gather(build_lut(histo, step), im))
#
#         return tf.cast(result, tf.uint8)
#
#     # Assumes RGB for now.  Scales each channel independently
#     # and then stacks the result.
#     s1 = scale_channel(image, 0)
#     s2 = scale_channel(image, 1)
#     s3 = scale_channel(image, 2)
#     image = tf.stack([s1, s2, s3], 2)
#     return image
#
#
# def sharpness(image, factor):
#     """Implements Sharpness function from PIL using TF ops."""
#     orig_image = image
#     image = tf.cast(image, tf.float32)
#     # Make image 4D for conv operation.
#     image = tf.expand_dims(image, 0)
#     # SMOOTH PIL Kernel.
#     kernel = tf.constant(
#         [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
#         shape=[3, 3, 1, 1]) / 13.
#     # Tile across channel dimension.
#     kernel = tf.tile(kernel, [1, 1, 3, 1])
#     strides = [1, 1, 1, 1]
#     degenerate = tf.nn.depthwise_conv2d(
#         image, kernel, strides, padding='VALID', dilations=[1, 1])
#     degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
#     degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])
#
#     # For the borders of the resulting image, fill in the values of the
#     # original image.
#     mask = tf.ones_like(degenerate)
#     padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
#     padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
#     result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)
#
#     # Blend the final result.
#     return blend(result, orig_image, factor)
#
#
# def shear_x(image, magnitude, replace):
#     image = transform(
#         wrap(image), [1., magnitude, 0., 0., 1., 0., 0., 0.])
#     return unwrap(image, replace)
#
#
# def shear_y(image, magnitude, replace):
#     image = transform(
#         wrap(image), [1., 0., 0., magnitude, 1., 0., 0., 0.])
#     return unwrap(image, replace)
#
#
# def translate_x(image, pixels, replace):
#     transforms = translations_to_projective_transforms([pixels, 0])
#     image = transform(wrap(image), transforms)
#     return unwrap(image, replace)
#
#
# def translate_y(image, pixels, replace):
#     transforms = translations_to_projective_transforms([0, pixels])
#     image = transform(wrap(image), transforms)
#     return unwrap(image, replace)
