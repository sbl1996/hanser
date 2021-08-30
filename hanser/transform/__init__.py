import math

from toolz import curry

import tensorflow as tf

from tensorflow_addons.image.translate_ops import translations_to_projective_transforms
from tensorflow_addons.image.transform_ops import angles_to_projective_transforms

from hanser.ops import log_uniform, prepend_dims
from hanser.transform.common import image_dimensions, to_4D_image, get_ndims, from_4D_image
from hanser.transform.mix import mixup, mixup_in_batch, mixup_batch, cutmix, cutmix_in_batch, cutmix_batch, fmix, mixup_cutmix_batch, mixup_or_cutmix_batch, mixup_cutmix_batch2, resizemix_batch


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


def random_apply(func, p, image):
    return tf.cond(
        tf.random.uniform(()) < p,
        lambda: func(image),
        lambda: image,
    )


def random_apply2(func, p, input1, input2):
    return tf.cond(
        tf.random.uniform(()) < p,
        lambda: func(input1, input2),
        lambda: (input1, input2),
    )


def random_choice(funcs, image):
    """Select a random policy from `policies` and apply it to `image`."""

    funcs_to_select = tf.random.uniform((), maxval=len(funcs), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(funcs):
        image = tf.cond(
            tf.equal(i, funcs_to_select),
            lambda: policy(image),
            lambda: image)
    return image


def resize(img, size, method='bilinear'):
    if img.dtype == tf.string:
        img = tf.image.decode_jpeg(img, channels=3)
    if not isinstance(size, (tuple, list)):
        size = tf.cast(size, tf.float32)
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        shorter = tf.minimum(h, w)
        scale = size / shorter
        oh = tf.cast(tf.math.ceil(h * scale), tf.int32)
        ow = tf.cast(tf.math.ceil(w * scale), tf.int32)
        size = tf.stack([oh, ow])
    dtype = img.dtype
    img = tf.image.resize(img, size, method=method)
    if img.dtype != dtype:
        img = tf.cast(img, dtype)
    return img


def random_resized_crop(image, size, scale=(0.05, 1.0), ratio=(0.75, 1.33)):
    bbox = tf.zeros((1, 0, 4), dtype=tf.float32)
    decoded = image.dtype != tf.string
    shape = tf.shape(image) if decoded else tf.image.extract_jpeg_shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        aspect_ratio_range=ratio,
        area_range=scale,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    if decoded:
        cropped = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_y,
            offset_width=offset_x,
            target_height=target_height,
            target_width=target_width)
    else:
        cropped = tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)

    if not isinstance(size, (tuple, list)):
        size = (size, size)
    image = resize(cropped, size)
    return image


def pad(image, padding, fill=0):
    if isinstance(padding, int):
        padding = (padding, padding)
    ph, pw = padding
    original_ndims = get_ndims(image)
    image = to_4D_image(image)

    paddings = [(0, 0), (ph, ph), (pw, pw), (0, 0)]
    image = tf.pad(image, paddings, constant_values=fill)
    return from_4D_image(image, original_ndims)


def random_crop(image, size, padding, fill=0):
    height, width = size
    original_ndims = get_ndims(image)
    image = to_4D_image(image)
    n, _h, _w, c = image_dimensions(image, 4)

    image = pad(image, padding, fill)
    crop_size = [n, height, width, c]
    image = tf.image.random_crop(image, crop_size)
    return from_4D_image(image, original_ndims)


def center_crop(image, size):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    crop_height, crop_width = size

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


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
def cutout(image, length, fill=0):
    r"""Cutout, support batch input.
    Args:
        image: (N?, H, W, C) single image or a batch of images.
        length: length of the cutout region.
        fill: value to be filled in cutout region, maybe scalar, tensor, 'normal' and 'uniform'.
    Notes:
        Set `fill` to 'normal' only if pixel values are in [0, 1], and 'uniform' if in [0, 255].
    """
    original_ndims = get_ndims(image)
    image = to_4D_image(image)

    bs, h, w, c = image_dimensions(image, 4)
    cy = tf.random.uniform((bs,), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((bs,), 0, w, dtype=tf.int32)

    length = length // 2
    t = tf.maximum(0, cy - length)
    b = tf.minimum(h, cy + length)
    l = tf.maximum(0, cx - length)
    r = tf.minimum(w, cx + length)

    hi = tf.range(h)[None, :, None, None]
    mh = (hi >= t[:, None, None, None]) & (hi < b[:, None, None, None])
    wi = tf.range(w)[None, None, :, None]
    mw = (wi >= l[:, None, None, None]) & (wi < r[:, None, None, None])
    masks = tf.cast(tf.logical_not(mh & mw), image.dtype)

    fill = _fill_region((bs, h, w, c), fill, image.dtype)
    image = tf.where(tf.equal(masks, 0), fill, image)
    image = from_4D_image(image, original_ndims)
    return image


@curry
def cutout2(image, length, fill=0):
    height, width, channels = image_dimensions(image, 3)

    center_y = tf.random.uniform((), 0, height, dtype=tf.int32)
    center_x = tf.random.uniform((), 0, width, dtype=tf.int32)

    length = length // 2
    lower_pad = tf.maximum(0, center_y - length)
    upper_pad = tf.maximum(0, height - center_y - length)
    left_pad = tf.maximum(0, center_x - length)
    right_pad = tf.maximum(0, width - center_x - length)

    cutout_shape = [height - (lower_pad + upper_pad),
                    width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.tile(mask[:, :, None], [1, 1, channels])
    fill = _fill_region((height, width, channels), fill, image.dtype)
    image = tf.where(tf.equal(mask, 0), fill, image)
    image.set_shape((height, width, channels))
    return image


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


# noinspection PyUnboundLocalVariable
@curry
def random_erasing(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=None, fill='normal'):
    r"""Random Erasing.
    Args:
        image: the input image with shape (H, W, C), C may be 1, 3, 4.
        p: probabilty to apply this augmentation.
        s_l: lower bound of erasing area ratio.
        s_h: upper bound of erasing area ratio.
        r_1: lower bound of erasing aspect ratio.
        r_2: upper bound of erasing aspect ratio, default to 1/r_1.
        fill: value to be filled in erasing area, can be scalar, vector, 'normal' or 'uniform'.
    Notes:
        Set `fill` to 'normal' if pixel values are in [0, 1], and 'uniform' if in [0, 255].
    """
    if tf.random.uniform(()) > p:
        return image

    if r_2 is None:
        r_2 = 1 / r_1
    height, width, c = image_dimensions(image, 3)
    area = tf.cast(height * width, tf.float32)

    h = tf.zeros((), dtype=tf.int32)
    w = tf.zeros((), dtype=tf.int32)
    while tf.constant(True):
        target_area = tf.random.uniform((), s_l, s_h) * area
        aspect_ratio = log_uniform((), r_1, r_2)
        h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

        if h < height and w < width:
            break

    x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
    y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

    top = image[0:y, :, :]
    mid_left = image[y:y+h, 0:x, :]
    mid_right = image[y:y+h, x+w:width, :]
    bottom = image[y+h:height, :, :]

    fill = _fill_region((h, w, c), fill, image.dtype)
    mid = tf.concat([mid_left, fill, mid_right], 1)
    image = tf.concat([top, mid, bottom], 0)
    image.set_shape((height, width, c))
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
    temp = image1 + scaled

    # Interpolate
    # noinspection PyChainedComparisons
    if factor > 0.0 and factor < 1.0:
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
    threshold = tf.cast(threshold, image.dtype)
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int32) + addition
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
    mean = tf.minimum(mean, 255.0)
    degenerate = tf.fill(degenerate.shape, mean)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    bits = tf.cast(bits, image.dtype)
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, bits), bits)


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
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    transforms = angles_to_projective_transforms(
        radians, image_height, image_width)
    image = transform(wrap(image), transforms)
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
    alpha_channel = flattened_image[:, 3][:, None]

    replace = tf.constant(replace, tf.uint8)
    if tf.rank(replace) == 0:
        replace = tf.expand_dims(replace, 0)
        replace = tf.concat([replace, replace, replace], 0)
    replace = tf.concat([replace, tf.ones([1], dtype=image.dtype)], 0)

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

    def scale_channel(img):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(img), tf.float32)
        hi = tf.cast(tf.reduce_max(img), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(img), lambda: img)
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

        # noinspection PyShadowingNames
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
        image, kernel, strides, padding='VALID', dilations=[1, 1])
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
    image = transform(
        wrap(image), [1., magnitude, 0., 0., 1., 0., 0., 0.])
    return unwrap(image, replace)


def shear_y(image, magnitude, replace):
    image = transform(
        wrap(image), [1., 0., 0., magnitude, 1., 0., 0., 0.])
    return unwrap(image, replace)


def translate_x(image, pixels, replace):
    transforms = translations_to_projective_transforms([pixels, 0])
    image = transform(wrap(image), transforms)
    return unwrap(image, replace)


def translate_y(image, pixels, replace):
    transforms = translations_to_projective_transforms([0, pixels])
    image = transform(wrap(image), transforms)
    return unwrap(image, replace)


def normalize(x, mean, std):
    mean = tf.convert_to_tensor(mean, x.dtype)
    std = tf.convert_to_tensor(std, x.dtype)
    x = (x - mean) / std
    return x


def to_tensor(image, label, dtype=tf.float32, vmax=255, label_offset=None):
    image = tf.cast(image, dtype) / vmax
    label = tf.reshape(label, shape=())
    label = tf.cast(label, tf.int32)
    if label_offset:
        label -= label_offset
    return image, label


def photo_metric_distortion(image,
                            brightness_delta=32,
                            contrast_range=(0.5, 1.5),
                            saturation_range=(0.5, 1.5),
                            hue_delta=18):
    dtype = image.dtype
    image = tf.cast(image, tf.float32) / 255

    if tf.random.uniform(()) < 0.5:
        image = tf.clip_by_value(tf.image.random_brightness(image, brightness_delta / 255), 0, 1)

    mode = tf.random.uniform((), 0, 2, dtype=tf.int32)

    if mode == 0 and tf.random.uniform(()) < 0.5:
        image = tf.clip_by_value(tf.image.random_contrast(image, contrast_range[0], contrast_range[1]), 0, 1)

    if tf.random.uniform(()) < 0.5:
        image = tf.clip_by_value(tf.image.random_saturation(image, saturation_range[0], saturation_range[1]), 0, 1)

    if tf.random.uniform(()) < 0.5:
        image = tf.clip_by_value(tf.image.random_hue(image, hue_delta / 255), 0, 1)

    if mode == 1 and tf.random.uniform(()) < 0.5:
        image = tf.clip_by_value(tf.image.random_contrast(image, contrast_range[0], contrast_range[1]), 0, 1)

    image = tf.cast(image * 255, dtype)

    return image


# noinspection PyShadowingNames
def color_jitter(image, brightness, contrast, saturation, hue):
    dtype = image.dtype
    image = tf.cast(image, tf.float32) / 255

    order = tf.random.shuffle(tf.range(4))
    for i in order:
        if i == 0:
            if brightness != 0 and tf.random.uniform(()) < 0.5:
                image = tf.clip_by_value(tf.image.random_brightness(image, brightness), 0, 1)
        elif i == 1:
            if contrast != 0 and tf.random.uniform(()) < 0.5:
                image = tf.clip_by_value(tf.image.random_contrast(image, 1 - contrast, 1 + contrast), 0, 1)
        elif i == 2:
            if saturation != 0 and tf.random.uniform(()) < 0.5:
                image = tf.clip_by_value(tf.image.random_saturation(image, 1 - saturation, 1 + saturation), 0, 1)
        else:
            if hue != 0 and tf.random.uniform(()) < 0.5:
                image = tf.clip_by_value(tf.image.random_hue(image, hue), 0, 1)

    image = tf.cast(image * 255, dtype)

    return image


# noinspection PyShadowingNames
def color_jitter2(image, brightness, contrast, saturation, hue):
    image = tf.cast(image, tf.float32) / 255

    def branch_fn(i):
        def func(img):
            return tf.switch_case(i, [
                lambda: tf.clip_by_value(tf.image.random_brightness(img, brightness), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_contrast(img, 1 - contrast, 1 + contrast), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_saturation(img, 1 - saturation, 1 + saturation), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_hue(img, hue), 0, 1),
            ])

        return func

    order = tf.random.shuffle(tf.range(4))
    image = tf.while_loop(
        lambda i, im: i < 4,
        lambda i, im: [i + 1, random_apply(branch_fn(order[i]), 0.5, image)],
        [0, image],
    )[1]

    image = tf.cast(image * 255, tf.uint8)

    return image


_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


# noinspection PyDefaultArgument
def lighting(x, alpha_std, eig_val=_EIG_VALS, eig_vec=_EIG_VECS, vmax=255):
    """Performs AlexNet-style PCA jitter (used for training)."""
    eig_val = tf.convert_to_tensor(eig_val, x.dtype)
    eig_vec = tf.convert_to_tensor(eig_vec, x.dtype)
    alpha = tf.random.normal((1, 3), 0, alpha_std, x.dtype)
    rgb = tf.reduce_sum(eig_val * eig_vec * alpha, axis=1)
    rgb = rgb * tf.convert_to_tensor(vmax, x.dtype)
    x = x + rgb
    return x


def pad_to_bounding_box(
    image, offset_height, offset_width, target_height, target_width, pad_value):
    original_ndims = get_ndims(image)
    image = to_4D_image(image)
    batch, height, width, depth = image_dimensions(image, rank=4)

    ph1, ph2 = offset_height, target_height - offset_height - height
    pw1, pw2 = offset_width, target_width - offset_width - width

    image -= pad_value

    padded = tf.pad(image, [(0, 0), (ph1, ph2), (pw1, pw2), (0, 0)])
    outputs = padded + pad_value

    if isinstance(target_height, int):
        outputs.set_shape([outputs.shape[0], target_height, target_width, outputs.shape[-1]])
    outputs = from_4D_image(outputs, original_ndims)
    return outputs


def resize_longer(img, size, method='bilinear'):
    h, w, c = image_dimensions(img, 3)
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    longer = tf.maximum(h, w)
    scale = size / longer
    oh = tf.cast(tf.math.ceil(h * scale), tf.int32)
    ow = tf.cast(tf.math.ceil(w * scale), tf.int32)
    size = tf.stack([oh, ow])
    dtype = img.dtype
    img = tf.image.resize(img, size, method=method)
    if img.dtype != dtype:
        img = tf.cast(img, dtype)
    return img
