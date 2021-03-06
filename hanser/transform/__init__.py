import math

from toolz import curry

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.python.ops import gen_image_ops

from hanser.ops import beta_mc
from hanser.transform.fmix import sample_mask

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.120, 57.375]


def _mixup(image1, label1, image2, label2, lam):

    lam_image = tf.cast(lam, image1.dtype)[:, None, None, None]
    image = lam_image * image1 + (1 - lam_image) * image2

    lam_label = tf.cast(lam, label1.dtype)[:, None]
    label = lam_label * label1 + (1 - lam_label) * label2
    return image, label


def _get_lam(shape, alpha, uniform=False, mc=False):
    if uniform:
        lam = tf.random.uniform(shape)
    elif mc:
        lam = beta_mc(alpha, alpha, shape, mc_size=10000)
    else:
        lam = tfp.distributions.Beta(alpha, alpha).sample(shape)
    return lam


@curry
def mixup_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)
    return _mixup(image, label, image[::-1], label[::-1], lam)


@curry
def mixup_in_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0] // 2
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)
    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    return _mixup(image1, label1, image2, label2, lam)


@curry
def mixup(data1, data2, alpha, hard=False, **gen_lam_kwargs):
    image1, label1 = data1
    image2, label2 = data2

    is_batch = _is_batch(image1)
    image1, label1, image2, label2 = wrap_batch([
        image1, label1, image2, label2
    ], is_batch)

    n = _image_dimensions(image1, 4)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    image, label = _mixup(image1, label1, image2, label2, lam)

    image, label = unwrap_batch([image, label], is_batch)
    return image, label


def rand_bbox(h, w, lam):
    cut_rat = tf.sqrt(1. - lam)
    cut_w = tf.cast(tf.cast(w, lam.dtype) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(h, lam.dtype) * cut_rat, tf.int32)

    cx = tf.random.uniform(tf.shape(lam), 0, w, dtype=tf.int32)
    cy = tf.random.uniform(tf.shape(lam), 0, h, dtype=tf.int32)

    l = tf.clip_by_value(cx - cut_w // 2, 0, w)
    t = tf.clip_by_value(cy - cut_h // 2, 0, h)
    r = tf.clip_by_value(cx + cut_w // 2, 0, w)
    b = tf.clip_by_value(cy + cut_h // 2, 0, h)

    return l, t, r, b


def rand_mask(image, lam):
    n, h, w = _image_dimensions(image, 4)[:3]
    l, t, r, b = rand_bbox(h, w, lam)
    hi = tf.range(h)[None, :, None, None]
    mh = (hi >= t[:, None, None, None]) & (hi < b[:, None, None, None])
    wi = tf.range(w)[None, None, :, None]
    mw = (wi >= l[:, None, None, None]) & (wi < r[:, None, None, None])
    masks = tf.cast(tf.logical_not(mh & mw), image.dtype)
    lam = 1 - (b - t) * (r - l) / (h * w)
    return masks, lam


@curry
def cutmix_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = _image_dimensions(image, 4)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    masks, lam = rand_mask(image, lam)

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    image = image * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label.dtype)[:, None]
    label = label * lam + label2 * (1. - lam)
    return image, label


@curry
def cutmix_in_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0] // 2
    lam_shape = (n,) if hard else ()
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    masks, lam = rand_mask(image1, lam)

    image = image1 * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label.dtype)[:, None]
    label = label1 * lam + label2 * (1. - lam)
    return image, label


def wrap_batch(tensors, is_batch):
    return tuple(t if is_batch else t[None] for t in tensors)


def unwrap_batch(tensors, is_batch):
    return tuple(t if is_batch else t[0] for t in tensors)


@curry
def cutmix(data1, data2, alpha, hard=False, **gen_lam_kwargs):
    image1, label1 = data1
    image2, label2 = data2

    is_batch = _is_batch(image1)
    image1, label1, image2, label2 = wrap_batch([
        image1, label1, image2, label2
    ], is_batch)

    n = _image_dimensions(image1, 4)[0]
    lam_shape = (n,) if hard else ()
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    masks, lam = rand_mask(image1, lam)

    image = image1 * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label1.dtype)[:, None]
    label = label1 * lam + label2 * (1. - lam)

    image, label = unwrap_batch([image, label], is_batch)
    return image, label


@curry
def fmix(data1, data2, alpha, decay_power):
    image1, label1 = data1
    image2, label2 = data2
    shape = image1.shape[:2]
    lam, mask = sample_mask(alpha, decay_power, shape)
    image = mask * image1 + (1 - mask) * image2

    lam = tf.cast(lam, label1.dtype)
    label = lam * label1 + (1 - lam) * label2
    return image, label


def get_ndims(image):
    return image.get_shape().ndims or tf.rank(image)


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.

    Args:
      image: 2/3/4D tensor.

    Returns:
      4D tensor with the same type.
    """
    with tf.control_dependencies(
        [
            tf.debugging.assert_rank_in(
                image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
            )
        ]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D tensor.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D tensor with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


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

    output = gen_image_ops.image_projective_transform_v2(
        images,
        output_shape=output_shape,
        transforms=transforms,
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


def _image_dimensions(image, rank):
    """Returns the dimensions of an image tensor.

    Args:
        image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
        rank: The expected rank of the image

    Returns:
        A list of corresponding to the dimensions of the input image. Dimensions
        that are statically known are python integers, otherwise they are integer
        scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [
            s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
        ]


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
    bbox = tf.zeros((1,0,4), dtype=tf.float32)
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


def _is_batch(x):
    return len(x.shape) == 4


def pad(x, padding, fill=0):
    if isinstance(padding, int):
        padding = (padding, padding)
    ph, pw = padding
    paddings = [(ph, ph), (pw, pw), (0, 0)]
    if _is_batch(x):
        paddings = [(0, 0), *paddings]
    x = tf.pad(x, paddings, constant_values=fill)
    return x


def random_crop(x, size, padding, fill=0):
    height, width = size
    x = pad(x, padding, fill)
    crop_size = [height, width, x.shape[-1]]
    if _is_batch(x):
        crop_size = [tf.shape(x)[0], *crop_size]
    x = tf.image.random_crop(x, crop_size)
    return x


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


@curry
def cutout(images, length):
    is_batch = _is_batch(images)
    if not is_batch:
        images = images[None]

    bs, h, w, c = _image_dimensions(images, 4)
    cy = tf.random.uniform((bs,), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((bs,), 0, w, dtype=tf.int32)

    t = tf.maximum(0, cy - length // 2)
    b = tf.minimum(h, cy + length // 2)
    l = tf.maximum(0, cx - length // 2)
    r = tf.minimum(w, cx + length // 2)

    hi = tf.range(h)[None, :, None, None]
    mh = (hi >= t[:, None, None, None]) & (hi < b[:, None, None, None])
    wi = tf.range(w)[None, None, :, None]
    mw = (wi >= l[:, None, None, None]) & (wi < r[:, None, None, None])
    masks = tf.cast(tf.logical_not(mh & mw), images.dtype)

    images = images * masks

    if not is_batch:
        images = images[0]
    return images


def invert(image):
    return 255 - image


def blend(image1, image2, factor):

    def blend_fn(image1, image2):
        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        temp = image1 + scaled
        return tf.cond(
            tf.logical_and(tf.greater(factor, 0.0), tf.less(factor, 1.0)),
            lambda: tf.cast(temp, tf.uint8),
            lambda: tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)
        )

    return tf.cond(
        tf.equal(factor, 0.0),
        lambda: tf.identity(image1),
        lambda: tf.cond(
            tf.equal(factor, 1.0),
            lambda: tf.identity(image2),
            lambda: blend_fn(image1, image2)
        )
    )


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
    image_height = tf.cast(tf.shape(image)[0], tf.dtypes.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.dtypes.float32)
    transforms = tfa.image.transform_ops.angles_to_projective_transforms(
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
    transform_or_transforms = tf.convert_to_tensor(
        [1., magnitude, 0., 0., 1., 0., 0., 0.], dtype=tf.dtypes.float32)
    image = transform(wrap(image), transform_or_transforms)
    return unwrap(image, replace)


def shear_y(image, magnitude, replace):
    transform_or_transforms = tf.convert_to_tensor(
        [1., 0., 0., magnitude, 1., 0., 0., 0.], dtype=tf.dtypes.float32)
    image = transform(wrap(image), transform_or_transforms)
    return unwrap(image, replace)


def translate_x(image, pixels, replace):
    translations = tf.convert_to_tensor(
        [pixels, 0], name="translations", dtype=tf.dtypes.float32)
    transforms = tfa.image.translate_ops.translations_to_projective_transforms(translations)
    image = transform(wrap(image), transforms)
    return unwrap(image, replace)


def translate_y(image, pixels, replace):
    translations = tf.convert_to_tensor(
        [0, pixels], name="translations", dtype=tf.dtypes.float32)
    transforms = tfa.image.translate_ops.translations_to_projective_transforms(translations)
    image = transform(wrap(image), transforms)
    return unwrap(image, replace)


def normalize(x, mean, std):
    mean = tf.convert_to_tensor(mean, x.dtype)
    std = tf.convert_to_tensor(std, x.dtype)
    mean = tf.broadcast_to(mean, tf.shape(x))
    std = tf.broadcast_to(std, tf.shape(x))
    x = (x - mean) / std
    return x


def to_tensor(image, label, dtype=tf.float32, vmax=255, label_offset=None):
    image = tf.cast(image, dtype) / vmax
    label = tf.reshape(label, shape=())
    label = tf.cast(label, tf.int32)
    if label_offset:
        label -= label_offset
    return image, label


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


def color_jitter2(image, brightness, contrast, saturation, hue):
    image = tf.cast(image, tf.float32) / 255

    def branch_fn(i):
        def func(image):
            return tf.switch_case(i, [
                lambda: tf.clip_by_value(tf.image.random_brightness(image, brightness), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_contrast(image, 1 - contrast, 1 + contrast), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_saturation(image, 1 - saturation, 1 + saturation), 0, 1),
                lambda: tf.clip_by_value(tf.image.random_hue(image, hue), 0, 1),
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


def lighting(x, alpha_std, eig_val=_EIG_VALS, eig_vec=_EIG_VECS, vmax=255):
    """Performs AlexNet-style PCA jitter (used for training)."""
    eig_val = tf.convert_to_tensor(eig_val, x.dtype)
    eig_vec = tf.convert_to_tensor(eig_vec, x.dtype)
    alpha = tf.random.normal((1, 3), 0, alpha_std, x.dtype)
    rgb = tf.reduce_sum(eig_val * eig_vec * alpha, axis=1)
    rgb = rgb * tf.convert_to_tensor(vmax, x.dtype)
    x = x + rgb
    return x
