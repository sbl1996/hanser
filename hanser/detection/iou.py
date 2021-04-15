import math
import tensorflow as tf
from hanser.ops import l2_norm, get_shape


def bbox_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, offset=False, check=True):

    assert mode in ['iou', 'giou', 'diou', 'ciou'], f'Unsupported mode {mode}'
    use_diou = mode in ['diou', 'ciou']

    batch_shape = tf.shape(bboxes1)[:-2]
    rows = get_shape(bboxes1, -2)
    cols = get_shape(bboxes2, -2)

    if check:
        tf.debugging.assert_equal(tf.shape(bboxes1)[:-2], tf.shape(bboxes2)[:-2])
        if is_aligned:
            tf.debugging.assert_equal(rows, cols)

    # TODO: These lines did not work on TPU, so we must hava another `bbox_iou2` for loss function.
    if rows * cols == 0:
        if is_aligned:
            return tf.zeros(tf.concat([batch_shape, (rows,)], axis=0), dtype=bboxes1.dtype)
        else:
            return tf.zeros(tf.concat([batch_shape, (rows, cols)], axis=0), dtype=bboxes1.dtype)

    hw1, hw2 = bbox_size(bboxes1, offset=offset), bbox_size(bboxes2, offset=offset)
    area1 = hw1[..., 0] * hw1[..., 1]
    area2 = hw2[..., 0] * hw2[..., 1]

    if not is_aligned:
        bboxes1 = bboxes1[..., :, None, :]
        bboxes2 = bboxes2[..., None, :, :]
        area1 = area1[..., :, None]
        area2 = area2[..., None, :]

    hw = intersect_size(bboxes1, bboxes2, offset=offset)
    overlap = hw[..., 0] * hw[..., 1]

    union = area1 + area2 - overlap
    ious = tf.math.divide_no_nan(overlap, union)
    if mode == 'iou':
        return ious

    # calculate gious
    enclose_hw = union_size(bboxes1, bboxes2, offset=offset)

    if use_diou:
        yx1 = bbox_center(bboxes1, offset=offset)
        yx2 = bbox_center(bboxes2, offset=offset)
        diag = l2_norm(yx1 - yx2)
        enclosed_diag = l2_norm(enclose_hw)
        dious = ious - tf.math.divide_no_nan(diag, enclosed_diag)
        dious = tf.clip_by_value(dious, -1.0, 1.0)
        if mode == 'diou':
            return dious

        h1, w1 = hw1[..., 0], hw1[..., 1]
        h2, w2 = hw2[..., 0], hw2[..., 1]

        factor = tf.convert_to_tensor(4 / math.pi ** 2, tf.float32)
        atan1, atan2 = tf.atan(tf.math.divide_no_nan(w1, h1)), tf.atan(tf.math.divide_no_nan(w2, h2))

        if not is_aligned:
            atan1 = atan1[..., :, None]
            atan2 = atan2[..., None, :]

        v = factor * tf.square(atan1 - atan2)
        cious = dious - tf.math.divide_no_nan(v**2, 1 - ious + v)
        cious = tf.clip_by_value(cious, -1.0, 1.0)
        return cious

    enclose_area = enclose_hw[..., 0] * enclose_hw[..., 1]
    gious = ious - tf.math.divide_no_nan(enclose_area - union, enclose_area)
    return gious


def bbox_size(bboxes, offset=False):
    if offset:
        hw = bboxes[..., 2:] + bboxes[..., :2]
    else:
        hw = bboxes[..., 2:] - bboxes[..., :2]
    hw = tf.maximum(hw, 0)
    return hw


def bbox_center(bboxes, offset=False):
    if offset:
        yx = (-bboxes[..., :2] + bboxes[..., 2:]) / 2
    else:
        yx = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    return yx


def intersect_size(bboxes1, bboxes2, offset=False):
    if offset:
        tl = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
        br = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])
        hw = br + tl
    else:
        tl = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])
        br = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])
        hw = br - tl
    hw = tf.maximum(hw, 0)
    return hw


def union_size(bboxes1, bboxes2, offset=False):
    if offset:
        tl = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])
        br = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
        hw = br + tl
    else:
        tl = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
        br = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
        hw = br - tl
    hw = tf.maximum(hw, 0)
    return hw


def bbox_iou2(bboxes1, bboxes2, mode='iou', is_aligned=False, offset=False):

    assert mode in ['iou', 'giou', 'diou', 'ciou'], f'Unsupported mode {mode}'
    use_diou = mode in ['diou', 'ciou']

    hw1, hw2 = bbox_size(bboxes1, offset=offset), bbox_size(bboxes2, offset=offset)
    area1 = hw1[..., 0] * hw1[..., 1]
    area2 = hw2[..., 0] * hw2[..., 1]

    if not is_aligned:
        bboxes1 = bboxes1[..., :, None, :]
        bboxes2 = bboxes2[..., None, :, :]
        area1 = area1[..., :, None]
        area2 = area2[..., None, :]

    hw = intersect_size(bboxes1, bboxes2, offset=offset)
    overlap = hw[..., 0] * hw[..., 1]

    union = area1 + area2 - overlap
    ious = tf.math.divide_no_nan(overlap, union)
    if mode == 'iou':
        return ious

    # calculate gious
    enclose_hw = union_size(bboxes1, bboxes2, offset=offset)

    if use_diou:
        yx1 = bbox_center(bboxes1, offset=offset)
        yx2 = bbox_center(bboxes2, offset=offset)
        diag = l2_norm(yx1 - yx2)
        enclosed_diag = l2_norm(enclose_hw)
        dious = ious - tf.math.divide_no_nan(diag, enclosed_diag)
        if mode == 'diou':
            dious = tf.clip_by_value(dious, -1.0, 1.0)
            return dious

        h1, w1 = hw1[..., 0], hw1[..., 1]
        h2, w2 = hw2[..., 0], hw2[..., 1]

        factor = tf.convert_to_tensor(4 / math.pi ** 2, tf.float32)
        atan1, atan2 = tf.atan(tf.math.divide_no_nan(w1, h1)), tf.atan(tf.math.divide_no_nan(w2, h2))

        if not is_aligned:
            atan1 = atan1[..., :, None]
            atan2 = atan2[..., None, :]

        v = factor * tf.square(atan1 - atan2)
        alpha = tf.math.divide_no_nan(v, 1 - ious + v)
        cious = dious - tf.stop_gradient(alpha) * v
        cious = tf.clip_by_value(cious, -1.0, 1.0)
        return cious

    enclose_area = enclose_hw[..., 0] * enclose_hw[..., 1]
    gious = ious - tf.math.divide_no_nan(enclose_area - union, enclose_area)
    return gious
