import math
from toolz import curry

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops


@curry
def mixup_batch(image, label, beta):
    lam = tfp.distributions.Beta(beta, beta).sample(())
    index = tf.random.shuffle(tf.range(tf.shape(image)[0]))

    lam = tf.cast(lam, image.dtype)
    image = lam * image + (1 - lam) * tf.gather(image, index)

    lam = tf.cast(lam, label.dtype)
    label = lam * label + (1 - lam) * tf.gather(label, index)
    return image, label

@curry
def mixup(data1, data2, beta):
    image1, label1 = data1
    image2, label2 = data2
    lam = tfp.distributions.Beta(beta, beta).sample(())

    lam = tf.cast(lam, image1.dtype)
    image = lam * image1 + (1 - lam) * image2

    lam = tf.cast(lam, label1.dtype)
    label = lam * label1 + (1 - lam) * label2
    return image, label


def rand_bbox(h, w, lam):
    cut_rat = tf.sqrt(1. - lam)
    cut_w = tf.cast(tf.cast(w, lam.dtype) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(h, lam.dtype) * cut_rat, tf.int32)

    cx = tf.random.uniform((), 0, w, dtype=tf.int32)
    cy = tf.random.uniform((), 0, h, dtype=tf.int32)

    l = tf.clip_by_value(cx - cut_w // 2, 0, w)
    t = tf.clip_by_value(cy - cut_h // 2, 0, h)
    r = tf.clip_by_value(cx + cut_w // 2, 0, w)
    b = tf.clip_by_value(cy + cut_h // 2, 0, h)

    return l, t, r, b


@curry
def cutmix_batch(image, label, beta):
    lam = tfp.distributions.Beta(beta, beta).sample(())
    index = tf.random.shuffle(tf.range(tf.shape(image)[0]))

    shape = tf.shape(image)
    h = shape[1]
    w = shape[2]

    l, t, r, b = rand_bbox(h, w, lam)
    shape = [b - t, r - l]
    padding = [(t, h - b), (l, w - r)]

    mask = tf.pad(tf.zeros(shape, dtype=image.dtype), padding, constant_values=1)
    mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)

    image2 = tf.gather(image, index)
    label2 = tf.gather(label, index)
    image = image * mask + image2 * (1. - mask)

    lam = 1 - (b - t) * (r - l) / (h * w)
    lam = tf.cast(lam, label.dtype)
    label = label * lam + label2 * (1. - lam)

    return image, label


@curry
def cutmix(data1, data2, beta):
    image1, label1 = data1
    image2, label2 = data2
    lam = tfp.distributions.Beta(beta, beta).sample(())

    shape = tf.shape(image1)
    h = shape[0]
    w = shape[1]

    l, t, r, b = rand_bbox(h, w, lam)
    shape = [b - t, r - l]
    padding = [(t, h - b), (l, w - r)]

    mask = tf.pad(tf.zeros(shape, dtype=image1.dtype), padding, constant_values=1)
    mask = tf.expand_dims(mask, -1)

    image = image1 * mask + image2 * (1. - mask)

    lam = 1 - (b - t) * (r - l) / (h * w)
    lam = tf.cast(lam, label1.dtype)
    label = label1 * lam + label2 * (1. - lam)

    return image, label


CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image


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


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: decode_and_center_crop(image_bytes, image_size),
        lambda: tf.image.resize(image, [image_size, image_size], method='bicubic')
    )

    return image


def decode_and_center_crop(image_bytes, image_size):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method='bicubic')
    return image


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


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.

    Args:
      x: A python object to check.

    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (tf.Tensor, tf.Variable))


def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.

    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.

    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.

    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [tf.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def resolve_shape(tensor, rank=None):
    if rank is not None:
        shape = tensor.get_shape().with_rank(rank).as_list()
    else:
        shape = tensor.get_shape().as_list()

    if None in shape:
        shape_dynamic = tf.shape(tensor)
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = shape_dynamic[i]

    return shape


def resize(img, size, method='bilinear'):
    if not isinstance(size, (tuple, list)):
        size = tf.cast(size, tf.float32)
        h, w = img.shape[:2]
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        shorter = tf.minimum(h, w)
        scale = size / shorter
        oh = tf.cast(tf.math.ceil(h * scale), tf.int32)
        ow = tf.cast(tf.math.ceil(w * scale), tf.int32)
        size = tf.stack([oh, ow])
    img = tf.image.resize(img, size, method=method)
    return img


def _CheckAtLeast3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.

    Args:
      image: >= 3-D Tensor of size [*, height, width, depth]
      require_static: If `True`, requires that all dimensions of `image` are known
        and non-zero.

    Raises:
      ValueError: if image.shape is not a [>= 3] vector.

    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
      containing an assert op is returned.
    """
    try:
        if image.get_shape().ndims is None:
            image_shape = image.get_shape().with_rank(3)
        else:
            image_shape = image.get_shape().with_rank_at_least(3)
    except ValueError:
        raise ValueError("'image' must be at least three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError('\'image\' must be fully defined.')
    if any(x == 0 for x in image_shape):
        raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [
            check_ops.assert_positive(
                tf.shape(image),
                ["all dims of 'image.shape' "
                 'must be > 0.']),
            check_ops.assert_greater_equal(
                tf.rank(image),
                3,
                message="'image' must be at least three-dimensional.")
        ]
    else:
        return []


def _ImageDimensions(image, rank):
    """Returns the dimensions of an image tensor.

    Args:
      image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
      rank: The expected rank of the image

    Returns:
      A list of corresponding to the dimensions of the
      input image.  Dimensions that are statically known are python integers,
      otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
        return [
            s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
        ]


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
        image: 3-D tensor with shape [height, width, channels]
        offset_height: Number of rows of zeros to add on top.
        offset_width: Number of columns of zeros to add on the left.
        target_height: Height of output image.
        target_width: Width of output image.
        pad_value: Value to pad the image tensor with.

    Returns:
        3-D tensor of shape [target_height, target_width, channels].

    Raises:
        ValueError: If the shape of image is incompatible with the offset_* or
        target_* arguments.
    """
    with tf.name_scope('pad_to_bounding_box'):
        is_batch = True
        image_shape = image.get_shape()
        if image_shape.ndims == 3:
            is_batch = False
            image = tf.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = tf.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        batch, height, width, depth = _ImageDimensions(image, rank=4)

        after_padding_width = target_width - offset_width - width

        after_padding_height = target_height - offset_height - height

        assert_ops += _assert(offset_height >= 0, ValueError,
                              'offset_height must be >= 0')
        assert_ops += _assert(offset_width >= 0, ValueError,
                              'offset_width must be >= 0')
        assert_ops += _assert(after_padding_width >= 0, ValueError,
                              'width must be <= target - offset')
        assert_ops += _assert(after_padding_height >= 0, ValueError,
                              'height must be <= target - offset')
        # image = control_flow_ops.with_dependencies(assert_ops, image)
        with tf.control_dependencies(assert_ops):
            image -= pad_value

        # Do not pad on the depth dimensions.
        paddings = array_ops.reshape(
            array_ops.stack([
                0, 0, offset_height, after_padding_height, offset_width,
                after_padding_width, 0, 0
            ]), [4, 2])
        padded = array_ops.pad(image, paddings)

        padded_shape = [
            None if _is_tensor(i) else i
            for i in [batch, target_height, target_width, depth]
        ]
        padded.set_shape(padded_shape)

        if not is_batch:
            padded = array_ops.squeeze(padded, axis=[0])
        outputs = padded + pad_value
        return outputs


def pad(x, padding, fill=0):
    if isinstance(padding, int):
        padding = (padding, padding)
    ph, pw = padding
    x = tf.pad(x, [(ph, ph), (pw, pw), (0, 0)], constant_values=fill)
    return x


def random_crop(x, size, padding, fill=0):
    height, width = size
    x = pad(x, padding, fill)
    x = tf.image.random_crop(x, [height, width, x.shape[-1]])
    return x


def cutout(image, length):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    cy = tf.random.uniform((), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((), 0, w, dtype=tf.int32)

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


def invert(image):
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

    def blend_fn(image1, image2):
        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        temp = tf.cast(image1, tf.float32) + scaled
        return tf.cond(
            tf.logical_and(tf.greater(factor, 0.0), tf.less(factor, 1.0)),
            lambda: tf.cast(temp, tf.uint8),
            lambda: tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)
        )

    return tf.cond(
        tf.equal(factor, 0.0),
        lambda: tf.convert_to_tensor(image1),
        lambda: tf.cond(
            tf.equal(factor, 1.0),
            lambda: tf.convert_to_tensor(image2),
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
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    transform_or_transforms = tf.convert_to_tensor(
        [1., magnitude, 0., 0., 1., 0., 0., 0.], dtype=tf.dtypes.float32)
    image = transform(wrap(image), transform_or_transforms)
    return unwrap(image, replace)


def shear_y(image, magnitude, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
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


def normalize(image, mean, std):
    mean = tf.convert_to_tensor(mean, image.dtype)
    std = tf.convert_to_tensor(std, image.dtype)
    image = (image - mean) / std
    return image


def to_tensor(image, label, dtype=tf.float32):
    image = tf.cast(image, dtype) / 255
    label = tf.cast(label, tf.int32)

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
