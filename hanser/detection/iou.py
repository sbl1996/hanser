import tensorflow as tf


def bbox_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = tf.constant([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ], dtype=tf.float32)
        >>> bboxes2 = tf.constant([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ], dtype=tf.float32)
        >>> overlaps = bbox_iou(bboxes1, bboxes2)
        >>> assert overlaps.shape.as_list() == [3, 3]
        >>> overlaps = bbox_iou(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape.as_list() == [3]

    Example:
        >>> empty = tf.zeros((0, 4))
        >>> nonempty = tf.constant([[0, 0, 10, 9]], dtype=tf.float32)
        >>> assert tuple(bbox_iou(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_iou(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_iou(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    tf.debugging.assert_equal(tf.shape(bboxes1)[:-2], tf.shape(bboxes2)[:-2])
    batch_shape = tf.shape(bboxes1)[:-2]

    rows = tf.shape(bboxes1)[-2]
    cols = tf.shape(bboxes2)[-2]
    if is_aligned:
        tf.debugging.assert_equal(rows, cols)

    if rows * cols == 0:
        if is_aligned:
            return tf.zeros(tf.concat([batch_shape, (rows,)], axis=0), dtype=bboxes1.dtype)
        else:
            return tf.zeros(tf.concat([batch_shape, (rows, cols)], axis=0), dtype=bboxes1.dtype)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = tf.maximum(rb - lt, 0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        union = area1 + area2 - overlap
        if mode == 'giou':
            enclosed_lt = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = tf.maximum(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = tf.minimum(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = tf.maximum(rb - lt, 0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = tf.minimum(bboxes1[..., :, None, :2],
                                     bboxes2[..., None, :, :2])
            enclosed_rb = tf.maximum(bboxes1[..., :, None, 2:],
                                     bboxes2[..., None, :, 2:])

    eps = tf.constant([eps], dtype=union.dtype)
    union = tf.maximum(union, eps)
    ious = overlap / union
    if mode == 'iou':
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = tf.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
