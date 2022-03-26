import tensorflow as tf

from hanser.ops import to_float, to_int
from hanser.transform import pad_to_bounding_box as pad_to_bounding_box_image, image_dimensions


def random_apply(func, p, *args):
    if tf.random.normal(()) < p:
        return func(*args)
    return args


def hflip(image, objects):
    image = tf.image.flip_left_right(image)
    bboxes = objects['gt_bbox']
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes_y = bboxes[..., 0]
    bboxes_x = bboxes[..., 1]
    bboxes_x = (1 - bboxes_x)[..., ::-1]
    bboxes = tf.reshape(tf.stack([bboxes_y, bboxes_x], axis=-1), [-1, 4])
    objects = update_bbox(objects, bboxes)
    return image, objects


def random_hflip(image, objects, p=0.5):
    return random_apply(hflip, p, image, objects)


def random_resize(image, size, ratio_range=(0.8, 1,2)):
    scale = tf.random.uniform((), minval=ratio_range[0], maxval=ratio_range[1], dtype=tf.float32)
    scaled_size = size * scale
    return resize(image, scaled_size)


def resize(image, size, keep_ratio=True):
    if keep_ratio:
        im_size = to_float(tf.shape(image)[:2])
        scale = tf.reduce_min(to_float(size) / im_size)
        scaled_size = to_int(im_size * scale)
    else:
        scaled_size = size
    image = tf.image.resize(image, scaled_size)
    return image


def update_bbox(objects, bboxes):
    return {
        **objects,
        'gt_bbox': bboxes,
    }


def map_bbox(objects, map_fn):
    return {
        **objects,
        'gt_bbox': map_fn(objects['gt_bbox']),
    }


def random_crop(image, objects, size):
    h, w = image_dimensions(image, 3)[:2]
    im_size = tf.shape(image)[:2]
    crop_size = tf.minimum(im_size, size)
    max_offset = tf.maximum(crop_size - im_size, 0) + 1
    offset_y = tf.random.uniform((), 0, max_offset[0], dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, max_offset[1], dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_size[0], crop_size[1])
    bboxes = objects['gt_bbox']
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes = (bboxes * to_float([h, w]) - to_float([offset_y, offset_x])) / to_float(crop_size)
    bboxes = tf.reshape(bboxes, [-1, 4])
    objects = update_bbox(objects, bboxes)
    objects = filter_objects(objects)
    objects = map_bbox(objects, lambda b: tf.clip_by_value(b, 0, 1))
    return image, objects


def filter_objects(objects):
    bboxes = objects['gt_bbox']
    centers = (bboxes[..., :2] + bboxes[:, 2:]) / 2
    mask = tf.reduce_all((centers > 0) & (centers < 1), axis=-1)
    return {
        k: v[mask] for k, v in objects.items()
    }


def random_sample_crop(
        image, objects,
        min_ious=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
        aspect_ratio_range=(0.5, 2.0)):

    min_object_covered = tf.random.shuffle(tf.convert_to_tensor(min_ious))[0]
    if min_object_covered == 1.0:
        return image, objects

    ori_image = image
    ori_objects = {**objects}
    bboxes = objects['gt_bbox']
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
    objects = update_bbox(objects, bboxes)
    objects = filter_objects(objects)
    objects = map_bbox(objects, lambda b: tf.clip_by_value(b, 0, 1))
    if tf.shape(objects['gt_bbox'])[0] == 0:
        return ori_image, ori_objects
    return image, objects


def random_expand(image, objects, max_scale, pad_value, prob=0.5):
    if tf.random.normal(()) < prob:
        return image, objects
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    scale = tf.random.uniform((), 1.0, max_scale)
    new_h = to_int(to_float(h) * scale)
    new_w = to_int(to_float(w) * scale)

    offset_y = tf.random.uniform((), 0, new_h - h + 1, dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, new_w - w + 1, dtype=tf.int32)

    image, objects = pad_to_bounding_box(image, objects, offset_y, offset_x, new_h, new_w, pad_value)
    return image, objects


def pad_to(image, objects, size, pad_value=0, mode='corner'):
    assert mode in ['corner', 'center', 'random']
    if mode == 'corner':
        h, w = size[0], size[1]
        return pad_to_bounding_box(image, objects, 0, 0, h, w, pad_value)
    elif mode == 'center': # center
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        target_height, target_width = size[0], size[1]
        offset_h = (target_height - height) // 2
        offset_w = (target_width - width) // 2
        return pad_to_bounding_box(image, objects, offset_h, offset_w, target_height, target_width, pad_value)
    else: # random
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        target_height, target_width = size[0], size[1]
        offset_h = tf.random.uniform((), 0, target_height - height + 1, dtype=tf.int32)
        offset_w = tf.random.uniform((), 0, target_width - width + 1, dtype=tf.int32)
        return pad_to_bounding_box(image, objects, offset_h, offset_w, target_height, target_width, pad_value)


def pad_to_bounding_box(image, objects, offset_height, offset_width, target_height, target_width, pad_value=0):
    shape = tf.shape(image)
    height, width  = shape[0], shape[1]
    image = pad_to_bounding_box_image(image, offset_height, offset_width, target_height, target_width, pad_value)

    scale = to_float([target_height, target_width]) / to_float([height, width])
    offset = tf.cast([offset_height, offset_width], tf.float32) / tf.cast([target_height, target_width], tf.float32)

    bboxes = objects['gt_bbox']
    bboxes = tf.reshape(bboxes, [-1, 2, 2])
    bboxes = bboxes / scale + offset
    bboxes = tf.reshape(bboxes, [-1, 4])
    bboxes = tf.clip_by_value(bboxes, 0., 1.)
    objects = update_bbox(objects, bboxes)
    return image, objects


def pad_to_fixed_size(data, target_size, pad_value=0):
    left_shape = data.shape[1:]
    num_instances = tf.shape(data)[0]
    pad_length = target_size - num_instances
    paddings = tf.fill([pad_length, *left_shape], tf.cast(pad_value, data.dtype))
    padded_data = tf.concat([data, paddings], axis=0)
    return padded_data


def pad_objects(objects, target_size, pad_value=0):
    padded = {}
    for k, data in objects.items():
        left_shape = data.shape[1:]
        num_instances = tf.shape(data)[0]
        pad_length = target_size - num_instances
        paddings = tf.fill([pad_length, *left_shape], tf.cast(pad_value, data.dtype))
        padded_data = tf.concat([data, paddings], axis=0)
        padded[k] = padded_data
    return padded
