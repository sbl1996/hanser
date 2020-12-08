import tensorflow as tf
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import array_ops

from hanser.ops import to_float, to_int, choice
from hanser.transform import _image_dimensions


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
        batch, height, width, depth = _image_dimensions(image, rank=4)

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


def random_choice(funcs, *args):
    funcs_to_select = tf.random.uniform((), maxval=len(funcs), dtype=tf.int32)
    for (i, func) in enumerate(funcs):
        if funcs_to_select == i:
            args = func(*args)
    return args


# def random_choice(funcs, image, boxes, classes, is_difficults):
#     """Select a random policy from `policies` and apply it to `image`."""
#
#     funcs_to_select = tf.random.uniform((), maxval=len(funcs), dtype=tf.int32)
#     # Note that using tf.case instead of tf.conds would result in significantly
#     # larger graphs and would even break export for some larger policies.
#     for (i, func) in enumerate(funcs):
#         image, boxes, classes, is_difficults = tf.cond(
#             tf.equal(i, funcs_to_select),
#             lambda: func(image, boxes, classes, is_difficults),
#             lambda: (image, boxes, classes, is_difficults))
#     return image, boxes, classes, is_difficults


def random_apply(func, p, *args):
    if tf.random.normal(()) < p:
        return func(*args)
    return args


# def random_apply2(func, p, image, boxes, classes, is_difficults):
#     return tf.cond(
#         tf.random.normal(()) < p,
#         lambda: func(image, boxes, classes, is_difficults),
#         lambda: (image, boxes, classes, is_difficults)
#     )


def get_random_scale(height, width, output_size, scale_min, scale_max):
    random_scale_factor = tf.random.uniform((), scale_min, scale_max)
    scaled_size = to_int(random_scale_factor * output_size)

    max_size = to_float(tf.maximum(height, width))
    img_scale = to_float(scaled_size) / max_size

    scaled_height = to_int(to_float(height) * img_scale)
    scaled_width = to_int(to_float(width) * img_scale)

    offset_y = to_float(scaled_height - output_size)
    offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform((), 0, 1)
    offset_y = to_int(offset_y)

    offset_x = to_float(scaled_width - output_size)
    offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform((), 0, 1)
    offset_x = to_int(offset_x)

    return img_scale, scaled_height, scaled_width, offset_x, offset_y


def hflip(image, bboxes):
    image = tf.image.flip_left_right(image)
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes_y = bboxes[..., 0]
    bboxes_x = bboxes[..., 1]
    bboxes_x = (1 - bboxes_x)[..., ::-1]
    bboxes = tf.reshape(tf.stack([bboxes_y, bboxes_x], axis=-1), [-1, 4])
    return image, bboxes


def random_hflip(image, bboxes, p=0.5):
    return random_apply(hflip, p, image, bboxes)


def filter_bboxes(bboxes, classes, is_difficults):
    centers = (bboxes[..., :2] + bboxes[:, 2:]) / 2
    mask = tf.reduce_all((centers > 0) & (centers < 1), axis=-1)
    return bboxes[mask], classes[mask], is_difficults[mask]


def random_sample_crop(
        image, bboxes, classes, is_difficults,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        aspect_ratio_range=(0.5, 2.0)):
    ori_image = image
    ori_bboxes = bboxes
    ori_classes = classes
    ori_diff = is_difficults

    min_object_covered = choice(min_ious)
    begin, size, box = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bboxes[None],
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=(0.1, 1.0),
    )
    image = tf.slice(image, begin, size)
    yx1 = box[0, 0, :2]
    yx2 = box[0, 0, 2:]
    size = yx2 - yx1
    bboxes = tf.reshape((tf.reshape(bboxes, [-1, 2, 2]) - yx1) / size, [-1, 4])
    bboxes, classes, is_difficults = filter_bboxes(bboxes, classes, is_difficults)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    if tf.shape(bboxes)[0] == 0:
        return ori_image, ori_bboxes, ori_classes, ori_diff
    return image, bboxes, classes, is_difficults


# def random_sample_crop(
#         image, bboxes, classes, is_difficults,
#         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#         aspect_ratio_range=(0.5, 2.0)):
#     ori_image = image
#     ori_bboxes = bboxes
#     ori_classes = classes
#     ori_diff = is_difficults
#
#     min_object_covered = choice(min_ious)
#     begin, size, box = tf.image.sample_distorted_bounding_box(
#         tf.shape(image), bboxes[None],
#         min_object_covered=min_object_covered,
#         aspect_ratio_range=aspect_ratio_range,
#         area_range=(0.1, 1.0),
#     )
#     image = tf.slice(image, begin, size)
#     yx1 = box[0, 0, :2]
#     yx2 = box[0, 0, 2:]
#     size = yx2 - yx1
#     bboxes = tf.reshape((tf.reshape(bboxes, [-1, 2, 2]) - yx1) / size, [-1, 4])
#     bboxes, classes, is_difficults = filter_bboxes(bboxes, classes, is_difficults)
#     bboxes = tf.clip_by_value(bboxes, 0, 1)
#     return tf.cond(
#         tf.shape(bboxes)[0] == 0,
#         lambda: (ori_image, ori_bboxes, ori_classes, ori_diff),
#         lambda: (image, bboxes, classes, is_difficults)
#     )


def scale_box(boxes, scales):
    return tf.reshape(tf.reshape(boxes, [-1, 2, 2]) * scales, [-1, 4])


def expand(image, boxes, max_scale, mean):
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    scale = tf.random.uniform((), 1.0, max_scale)
    new_h = to_int(to_float(h) * scale)
    new_w = to_int(to_float(w) * scale)

    offset_y = tf.random.uniform((), 0, new_h - h + 1, dtype=tf.int32)
    # offset_y = new_h - h
    offset_x = tf.random.uniform((), 0, new_w - w + 1, dtype=tf.int32)
    # offset_x = new_w - w

    image = pad_to_bounding_box(image, offset_y, offset_x, new_h, new_w, mean)
    boxes = tf.reshape(boxes, [-1, 2, 2])
    boxes = boxes / scale + tf.cast([offset_y, offset_x], tf.float32) / tf.cast([new_h, new_w], tf.float32)
    boxes = tf.reshape(boxes, [-1, 4])
    return image, boxes


def resize_with_pad(image, boxes, target_height, target_width, pad_value):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    img_scale = tf.minimum(
        to_float(target_height) / to_float(height),
        to_float(target_width) / to_float(width),
    )
    scaled_height = to_int(to_float(height) * img_scale)
    scaled_width = to_int(to_float(width) * img_scale)
    boxes = scale_box(boxes, to_float([scaled_height, scaled_width]) / to_float([target_height, target_width]))
    image = tf.image.resize(image, (scaled_height, scaled_width))
    image = pad_to_bounding_box(image, 0, 0, target_height, target_width, pad_value)
    return image, boxes


def resize_and_crop_image(image, scaled_height, scaled_width, output_size, offset_x, offset_y):
    image = tf.compat.v1.image.resize_bilinear(image[None], (scaled_height, scaled_width), align_corners=True)[0]
    image = image[
            offset_y:offset_y + output_size,
            offset_x:offset_x + output_size,
            ]
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, output_size, output_size
    )
    return image


def resize_and_crop_boxes(boxes, classes, scaled_height, scaled_width, output_size, offset_x, offset_y):
    boxes = tf.reshape(boxes, [-1, 2, 2])
    boxes = boxes * to_float([scaled_height, scaled_width]) - to_float([offset_y, offset_x])
    boxes = tf.reshape(boxes, [-1, 4])
    boxes = tf.clip_by_value(boxes, 0, output_size)
    indices = (boxes[:, 0] != boxes[:, 2]) & (boxes[:, 1] != boxes[:, 3])
    boxes = boxes[indices]
    classes = classes[indices]
    return boxes, classes


#
# def draw_bboxes(images, bboxes):
#     rank = images.ndim
#     assert bboxes.ndim == rank - 1
#     if rank == 3:
#         images = images[None]
#         bboxes = bboxes[None]
#     images = tf.cast(images, tf.float32)
#     images = tf.image.draw_bounding_boxes(images, bboxes, random_colors(16))
#     images = tf.cast(images, tf.uint8).numpy()
#     if rank == 3:
#         images = images[0]
#     return images

def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
    max_num_instances = output_shape[0]
    data = tf.reshape(data, [-1, *output_shape[1:]])
    num_instances = tf.shape(data)[0]
    assert_length = tf.Assert(
        tf.less_equal(num_instances, max_num_instances), [num_instances])
    with tf.control_dependencies([assert_length]):
        pad_length = max_num_instances - num_instances
    paddings = tf.fill([pad_length, *output_shape[1:]], tf.cast(pad_value, data.dtype))
    padded_data = tf.concat([data, paddings], axis=0)
    padded_data = tf.reshape(padded_data, output_shape)
    return padded_data


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

def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)
