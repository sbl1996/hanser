import tensorflow as tf

from hanser.ops import to_float, to_int, choice
from hanser.transform import pad_to_bounding_box as pad_to_bounding_box_image, _image_dimensions


def random_apply(func, p, *args):
    if tf.random.normal(()) < p:
        return func(*args)
    return args


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


def random_resize(image, size, ratio_range=(0.8, 1,2)):
    scale = tf.random.uniform((), minval=ratio_range[0], maxval=ratio_range[1], dtype=tf.float32)
    scaled_size = size * scale
    return resize(image, scaled_size)


def resize(image, size):
    im_size = to_float(tf.shape(image)[:2])
    scale = tf.reduce_min(to_float(size) / im_size)
    scaled_size = to_int(im_size * scale)
    image = tf.image.resize(image, scaled_size)
    return image


def random_crop(image, bboxes, labels, is_difficults, size):
    h, w = _image_dimensions(image, 3)[:2]
    im_size = tf.shape(image)[:2]
    crop_size = tf.minimum(im_size, size)
    max_offset = tf.maximum(crop_size - im_size, 0) + 1
    offset_y = tf.random.uniform((), 0, max_offset[0], dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, max_offset[1], dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_size[0], crop_size[1])
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes = (bboxes * to_float([h, w]) - to_float([offset_y, offset_x])) / to_float(crop_size)
    bboxes = tf.reshape(bboxes, [-1, 4])
    bboxes, labels, is_difficults = filter_bboxes(bboxes, labels, is_difficults)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return image, bboxes


def filter_bboxes(bboxes, labels, is_difficults):
    centers = (bboxes[..., :2] + bboxes[:, 2:]) / 2
    mask = tf.reduce_all((centers > 0) & (centers < 1), axis=-1)
    return bboxes[mask], labels[mask], is_difficults[mask]


def random_sample_crop(
        image, bboxes, labels, is_difficults,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        aspect_ratio_range=(0.5, 2.0)):
    ori_image = image
    ori_bboxes = bboxes
    ori_classes = labels
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
    bboxes, labels, is_difficults = filter_bboxes(bboxes, labels, is_difficults)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    if tf.shape(bboxes)[0] == 0:
        return ori_image, ori_bboxes, ori_classes, ori_diff
    return image, bboxes, labels, is_difficults



def scale_bbox(bboxes, scales):
    return tf.reshape(tf.reshape(bboxes, [-1, 2, 2]) * scales, [-1, 4])


def random_expand(image, bboxes, max_scale, pad_value):
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    scale = tf.random.uniform((), 1.0, max_scale)
    new_h = to_int(to_float(h) * scale)
    new_w = to_int(to_float(w) * scale)

    offset_y = tf.random.uniform((), 0, new_h - h + 1, dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, new_w - w + 1, dtype=tf.int32)

    image, bboxes = pad_to_bounding_box(image, bboxes, offset_y, offset_x, new_h, new_w, pad_value)
    return image, bboxes


def pad_to(image, bboxes, size, pad_value=0):
    h, w = size[0], size[1]
    return pad_to_bounding_box(image, bboxes, 0, 0, h, w, pad_value)


def pad_to_bounding_box(image, bboxes, offset_height, offset_width, target_height, target_width, pad_value=0):
    shape = tf.shape(image)
    height, width  = shape[0], shape[1]
    image = pad_to_bounding_box_image(image, offset_height, offset_width, target_height, target_width, pad_value)

    scale = to_float([target_height, target_width]) / to_float([height, width])
    offset = tf.cast([offset_height, offset_width], tf.float32) / tf.cast([target_height, target_width], tf.float32)
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes = bboxes / scale + offset
    bboxes = tf.reshape(bboxes, [-1, 4])
    bboxes = tf.clip_by_value(bboxes, 0., 1.)
    return image, bboxes


def pad_to_fixed_size(data, pad_value, output_shape):
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
