import tensorflow as tf
from hanser.detection import bbox_iou
from hanser.ops import nonzero, triu

#
# def multiclass_nms(bboxes,
#                    scores,
#                    iou_threshold=0.5,
#                    score_threshold=0.01,
#                    topk=-1,
#                    return_inds=False):
#     """NMS for multi-class bboxes.
#
#     Args:
#         bboxes (Tensor): shape (n, q, 4)
#         scores (Tensor): shape (n, #class)
#         iou_threshold (float): IoU threshold to be considered as conflicted.
#         score_threshold (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         topk (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         return_inds (bool, optional): Whether return the indices of kept
#             bboxes. Default to False.
#
#     Returns:
#         tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
#             (k), and (k). Labels are 0-based.
#     """
#     num_bboxes, num_classes = scores.shape
#
#     if bboxes.shape[1] == 1:
#         bboxes = tf.tile(bboxes, (1, num_classes, 1))
#
#     labels = tf.arange(num_classes, dtype=tf.int32)
#     labels = tf.tile(labels[None], (num_bboxes, 1))
#
#     bboxes = tf.reshape(bboxes, (-1, 4))
#     scores = tf.reshape(scores, -1)
#     labels = tf.reshape(labels, -1)
#
#     valid_mask = scores > score_threshold
#     inds = nonzero(valid_mask)
#     bboxes = tf.gather(bboxes, inds)
#     scores = tf.gather(scores, inds)
#     labels = tf.gather(labels, inds)
#
#     if bboxes.numel() == 0:
#         if return_inds:
#             return bboxes, labels, inds
#         else:
#             return bboxes, labels
#
#     dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
#
#     if max_num > 0:
#         dets = dets[:max_num]
#         keep = keep[:max_num]
#
#     if return_inds:
#         return dets, labels[keep], keep
#     else:
#         return dets, labels[keep]


def fast_nms(bboxes,
             scores,
             iou_threshold=0.5,
             score_threshold=0.05,
             max_per_class=200,
             top_k=100):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        bboxes (Tensor): shape (n, q, 4)
        scores (Tensor): shape (n, #class)
        iou_threshold (float): IoU threshold to be considered as conflicted.
        score_threshold (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        max_per_class (int): if there are more than top_k bboxes before NMS,
            only top max_per_class will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    num_bboxes, num_classes = scores.shape

    if bboxes.shape[1] == 1:
        bboxes = tf.tile(bboxes, (1, num_classes, 1))
    bboxes = tf.transpose(bboxes, (1, 0, 2))

    scores = tf.transpose(scores)  # [#class, n]
    scores, idx = tf.math.top_k(scores, max_per_class)
    bboxes = tf.gather(bboxes, idx, axis=1, batch_dims=1)

    ious = bbox_iou(bboxes, bboxes)  # [#class, topk, topk]
    ious = triu(ious, diag=False)
    keep = tf.reduce_max(ious, axis=1) <= iou_threshold
    keep = keep & (scores > score_threshold)

    labels = tf.range(num_classes, dtype=tf.int32)
    labels = tf.tile(labels[:, None], (1, max_per_class))
    labels = labels[keep]

    bboxes = bboxes[keep]
    scores = scores[keep]

    if tf.shape(bboxes)[0] > top_k:
        scores, idx = tf.math.top_k(scores, top_k)
        bboxes = bboxes[idx]
        labels = labels[idx]

    return bboxes, scores, labels


def soft_nms(bboxes,
             scores,
             score_threshold=0.05,
             sigma=0.5,
             max_per_class=200,
             top_k=100):
    """Soft NMS.

    Args:
        bboxes (Tensor): shape (n, q, 4)
        scores (Tensor): shape (n, #class)
        score_threshold (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        sigma: A `Tensor`. Must have the same type as `boxes`.
          A 0-D float tensor representing the sigma parameter for Soft NMS; see Bodla et
          al (c.f. https://arxiv.org/abs/1704.04503).  When `soft_nms_sigma=0.0` (which
          is default), we fall back to standard (hard) NMS.
        max_per_class (int): if there are more than top_k bboxes before NMS,
            only top max_per_class will be kept.
        top_k (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    num_bboxes, num_classes = scores.shape

    if bboxes.shape[1] == 1:
        bboxes = tf.tile(bboxes, (1, num_classes, 1))
    bboxes = tf.transpose(bboxes, (1, 0, 2))

    scores = tf.transpose(scores)  # [#class, n]
    scores, idx = tf.math.top_k(scores, max_per_class)
    bboxes = tf.gather(bboxes, idx, axis=1, batch_dims=1)

    labels = tf.range(num_classes, dtype=tf.int32)
    labels = tf.tile(labels[:, None], (1, max_per_class))

    bboxes = tf.reshape(bboxes, (-1, 4))
    scores = tf.reshape(scores, -1)
    labels = tf.reshape(labels, -1)

    idx, scores, n_valids = tf.raw_ops.NonMaxSuppressionV5(
            bboxes, scores, top_k, 1.0, score_threshold, sigma / 2, pad_to_max_output_size=True)

    bboxes = tf.gather(bboxes, idx)
    labels = tf.gather(labels, idx)
    return bboxes, scores, labels, n_valids
