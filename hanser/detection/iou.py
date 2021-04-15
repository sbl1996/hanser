import math
import tensorflow as tf
from hanser.ops import l2_norm, get_shape


# def bbox_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
#     """Calculate overlap between two set of bboxes.
#
#     If ``is_aligned `` is ``False``, then calculate the overlaps between each
#     bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
#     pair of bboxes1 and bboxes2.
#
#     Args:
#         bboxes1 (Tensor): shape (B, m, 4) in <y1, x1, y2, x2> format or empty.
#         bboxes2 (Tensor): shape (B, n, 4) in <y1, x1, y2, x2> format or empty.
#             B indicates the batch dim, in shape (B1, B2, ..., Bn).
#             If ``is_aligned `` is ``True``, then m and n must be equal.
#         mode (str): "iou" (intersection over union), "iof" (intersection over
#             foreground) or "giou" (generalized intersection over union).
#             Default "iou".
#         is_aligned (bool, optional): If True, then m and n must be equal.
#             Default False.
#         eps (float, optional): A value added to the denominator for numerical
#             stability. Default 1e-6.
#
#     Returns:
#         Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
#
#     Example:
#         >>> bboxes1 = tf.constant([
#         >>>     [0, 0, 10, 10],
#         >>>     [10, 10, 20, 20],
#         >>>     [32, 32, 38, 42],
#         >>> ], dtype=tf.float32)
#         >>> bboxes2 = tf.constant([
#         >>>     [0, 0, 10, 20],
#         >>>     [0, 10, 10, 19],
#         >>>     [10, 10, 20, 20],
#         >>> ], dtype=tf.float32)
#         >>> overlaps = bbox_iou(bboxes1, bboxes2)
#         >>> assert overlaps.shape.as_list() == [3, 3]
#         >>> overlaps = bbox_iou(bboxes1, bboxes2, is_aligned=True)
#         >>> assert overlaps.shape.as_list() == [3]
#
#     Example:
#         >>> empty = tf.zeros((0, 4))
#         >>> nonempty = tf.constant([[0, 0, 10, 9]], dtype=tf.float32)
#         >>> assert tuple(bbox_iou(empty, nonempty).shape) == (0, 1)
#         >>> assert tuple(bbox_iou(nonempty, empty).shape) == (1, 0)
#         >>> assert tuple(bbox_iou(empty, empty).shape) == (0, 0)
#     """
#
#     assert mode in ['iou', 'giou', 'diou', 'ciou'], f'Unsupported mode {mode}'
#     # Either the boxes are empty or the length of boxes' last dimension is 4
#     assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
#     assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)
#
#     use_enclosed = mode in ['giou', 'diou', 'ciou']
#     use_diou = mode in ['diou', 'ciou']
#
#     # Batch dim must be the same
#     # Batch dim: (B1, B2, ... Bn)
#     tf.debugging.assert_equal(tf.shape(bboxes1)[:-2], tf.shape(bboxes2)[:-2])
#     batch_shape = tf.shape(bboxes1)[:-2]
#
#     rows = tf.shape(bboxes1)[-2]
#     cols = tf.shape(bboxes2)[-2]
#     if is_aligned:
#         tf.debugging.assert_equal(rows, cols)
#
#     if rows * cols == 0:
#         if is_aligned:
#             return tf.zeros(tf.concat([batch_shape, (rows,)], axis=0), dtype=bboxes1.dtype)
#         else:
#             return tf.zeros(tf.concat([batch_shape, (rows, cols)], axis=0), dtype=bboxes1.dtype)
#
#     hw1 = bboxes1[..., 2:] - bboxes1[..., :2]
#     hw2 = bboxes2[..., 2:] - bboxes2[..., :2]
#
#     if use_diou:
#         yx1 = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
#         yx2 = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
#
#     area1 = hw1[..., 0] * hw1[..., 1]
#     area2 = hw2[..., 0] * hw2[..., 1]
#
#     if is_aligned:
#         tl = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
#         br = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
#
#         hw = tf.maximum(br - tl, 0)  # [B, rows, 2]
#         overlap = hw[..., 0] * hw[..., 1]
#
#         if use_diou:
#             diag = l2_norm(yx1 - yx2)
#
#         union = area1 + area2 - overlap
#         if use_enclosed:
#             enclosed_tl = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
#             enclosed_br = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
#     else:
#         tl = tf.maximum(bboxes1[..., :, None, :2],
#                         bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
#         br = tf.minimum(bboxes1[..., :, None, 2:],
#                         bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
#
#         hw = tf.maximum(br - tl, 0)  # [B, rows, cols, 2]
#         overlap = hw[..., 0] * hw[..., 1]
#
#         if use_diou:
#             diag = l2_norm(yx1[..., :, None, :] - yx2[..., None, :, :])
#
#         union = area1[..., None] + area2[..., None, :] - overlap
#         if use_enclosed:
#             enclosed_tl = tf.minimum(bboxes1[..., :, None, :2],
#                                      bboxes2[..., None, :, :2])
#             enclosed_br = tf.maximum(bboxes1[..., :, None, 2:],
#                                      bboxes2[..., None, :, 2:])
#
#     eps = tf.constant([eps], dtype=union.dtype)
#     union = tf.maximum(union, eps)
#     ious = overlap / union
#     if mode == 'iou':
#         return ious
#     # calculate gious
#     enclose_hw = tf.maximum(enclosed_br - enclosed_tl, 0)
#     if use_diou:
#         enclosed_diag = l2_norm(enclose_hw)
#         enclosed_diag = tf.maximum(enclosed_diag, eps)
#         dious = ious - diag / enclosed_diag
#         if mode == 'diou':
#             dious = tf.clip_by_value(dious, -1.0, 1.0)
#             return dious
#
#         h1, w1 = hw1[..., 0] + eps, hw1[..., 1]
#         h2, w2 = hw2[..., 0] + eps, hw2[..., 1]
#
#         factor = tf.convert_to_tensor(4 / math.pi ** 2, bboxes1.dtype)
#         if is_aligned:
#             v = factor * tf.square(tf.atan(w1 / h1) - tf.atan(w2 / h2))
#         else:
#             v = factor * tf.square(
#                 tf.atan(w1 / h1)[..., :, None] - tf.atan(w2 / h2)[..., None, :])
#         cious = dious - v**2 / (1 - ious + v)
#         cious = tf.clip_by_value(cious, -1.0, 1.0)
#         return cious
#     enclose_area = enclose_hw[..., 0] * enclose_hw[..., 1]
#     enclose_area = tf.maximum(enclose_area, eps)
#     gious = ious - (enclose_area - union) / enclose_area
#     return gious


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


def bbox_iou2(bboxes1, bboxes2, mode='iou', is_aligned=False, offset=False, check=True):

    assert mode in ['iou', 'giou', 'diou', 'ciou'], f'Unsupported mode {mode}'
    use_diou = mode in ['diou', 'ciou']

    batch_shape = tf.shape(bboxes1)[:-2]
    rows = get_shape(bboxes1, -2)
    cols = get_shape(bboxes2, -2)
    if check:
        tf.debugging.assert_equal(tf.shape(bboxes1)[:-2], tf.shape(bboxes2)[:-2])
        if is_aligned:
            tf.debugging.assert_equal(rows, cols)
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
