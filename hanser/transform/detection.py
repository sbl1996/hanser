import tensorflow as tf
from hanser.ops import to_float, to_int, choice
from hanser.transform import pad_to_bounding_box


def random_choice(funcs, image, boxes, classes):
    """Select a random policy from `policies` and apply it to `image`."""

    funcs_to_select = tf.random.uniform((), maxval=len(funcs), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, func) in enumerate(funcs):
        image, boxes = tf.cond(
            tf.equal(i, funcs_to_select),
            lambda: func(image, boxes, classes),
            lambda: (image, boxes, classes))
    return image, boxes


def random_apply(func, p, image, boxes, classes):
    return tf.cond(
        tf.random.normal(()) < p,
        lambda: func(image, boxes, classes),
        lambda: (image, boxes, classes)
    )


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


def hflip(image, bboxes, classes):
    image = tf.image.flip_left_right(image)
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes_y = bboxes[..., 0]
    bboxes_x = bboxes[..., 1]
    bboxes_x = (1 - bboxes_x)[..., ::-1]
    bboxes = tf.reshape(tf.stack([bboxes_y, bboxes_x], axis=-1), [-1, 4])
    return image, bboxes, classes


def random_hflip(image, bboxes, classes, p=0.5):
    return random_apply(hflip, p, image, bboxes, classes)


def filter_bboxes(bboxes, classes):
    centers = (bboxes[..., :2] + bboxes[:, 2:]) / 2
    mask = tf.reduce_all((centers > 0) & (centers < 1), axis=-1)
    return bboxes[mask], classes[mask]


def random_sample_crop(
        image, bboxes, classes,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        aspect_ratio_range=(0.5, 2.0)):
    ori_image = image
    ori_bboxes = bboxes
    ori_classes = classes

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
    bboxes, classes = filter_bboxes(bboxes, classes)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return tf.cond(
        tf.shape(bboxes)[0] == 0,
        lambda: (ori_image, ori_bboxes, ori_classes),
        lambda: (image, bboxes, classes)
    )


def scale_box(boxes, scales):
    return tf.reshape(tf.reshape(boxes, [-1, 2, 2]) * scales, [-1, 4])


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
    image = tf.image.resize(image, (scaled_height,  scaled_width))
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