import tensorflow as tf
from hanser.detection.iou import bbox_iou2
from hanser.ops import triu

def nms(bboxes, scores,
        iou_threshold=0.5,
        score_threshold=0.05,
        soft_nms_sigma=0.5,
        max_per_class=100,
        topk=200):
    """Non-max suppression.

    Args:
        bboxes (Tensor): shape (n, q, 4)
        scores (Tensor): shape (n, #class)
        iou_threshold (float): IoU threshold to be considered as conflicted.
        score_threshold (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        soft_nms_sigma: A `Tensor`. Must have the same type as `boxes`.
          A 0-D float tensor representing the sigma parameter for Soft NMS; see Bodla et
          al (c.f. https://arxiv.org/abs/1704.04503).  When `soft_nms_sigma=0.0` (which
          is default), we fall back to standard (hard) NMS.
        max_per_class (int): if there are more than top_k bboxes before NMS,
            only top max_per_class will be kept.
        topk (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, scores, labels, n_valids), tensors of shape (k, 4), (k,), (k,), scalar.
            Labels are 0-based.
    """
    if soft_nms_sigma != 0.0:
        iou_threshold = 1.0

    num_bboxes, num_classes = scores.shape

    if bboxes.shape[1] == 1:
        bboxes = tf.tile(bboxes, (1, num_classes, 1))
    bboxes = tf.transpose(bboxes, (1, 0, 2))

    scores = tf.transpose(scores)  # [#class, n]

    labels = tf.range(num_classes, dtype=tf.int32)
    labels = tf.tile(labels[:, None], (1, num_bboxes))

    all_bboxes = []
    all_scores = []
    all_labels = []
    all_n_valids = []
    for i in range(num_classes):
        bboxes_cls = bboxes[i]
        scores_cls = scores[i]
        labels_cls = labels[i]

        idx, scores_cls, n_valids = tf.raw_ops.NonMaxSuppressionV5(
                boxes=bboxes_cls, scores=scores_cls, max_output_size=max_per_class,
                iou_threshold=iou_threshold, score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma / 2, pad_to_max_output_size=False)
        all_bboxes.append(tf.gather(bboxes_cls, idx))
        all_scores.append(scores_cls)
        all_labels.append(tf.gather(labels_cls, idx))
        all_n_valids.append(n_valids)
    bboxes = tf.concat(all_bboxes, 0)
    scores = tf.concat(all_scores, 0)
    labels = tf.concat(all_labels, 0)
    n_valids = tf.reduce_sum(tf.stack(all_n_valids))
    if n_valids > topk:
        scores, idx = tf.math.top_k(scores, k=topk, sorted=True)
        bboxes = tf.gather(bboxes, idx)
        labels = tf.gather(labels, idx)
        n_valids = topk
    else:
        p = topk - n_valids
        bboxes = tf.pad(bboxes, [[0, p], [0, 0]])
        scores = tf.pad(scores, [[0, p]])
        labels = tf.pad(labels, [[0, p]])
    return bboxes, scores, labels, n_valids


@tf.function
def batched_nms_raw(
    bboxes, scores, iou_threshold, score_threshold, soft_nms_sigma, max_per_class, topk):

    batch_size = bboxes.shape[0]
    outputs = []
    for i in range(batch_size):
        outputs.append(
            nms(bboxes[i], scores[i], iou_threshold, score_threshold, soft_nms_sigma,
                max_per_class, topk))
    return [tf.stack(y) for y in zip(*outputs)]


def batched_nms(bboxes, scores, iou_threshold=0.5, score_threshold=0.05,
                soft_nms_sigma=0.5, max_per_class=100, topk=200):
    iou_threshold = tf.constant(iou_threshold, tf.float32)
    score_threshold = tf.constant(score_threshold, tf.float32)
    soft_nms_sigma = tf.constant(soft_nms_sigma, tf.float32)
    max_per_class = tf.constant(max_per_class, tf.int32)
    topk = tf.constant(topk, tf.int32)
    return batched_nms_raw(
        bboxes, scores, iou_threshold, score_threshold=score_threshold,
        soft_nms_sigma=soft_nms_sigma, max_per_class=max_per_class, topk=topk)


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

    ious = bbox_iou2(bboxes, bboxes)  # [#class, topk, topk]
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