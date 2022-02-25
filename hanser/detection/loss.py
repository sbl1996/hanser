from toolz import curry
import numpy as np

import tensorflow as tf

from hanser.losses import reduce_loss, focal_loss, l1_loss, smooth_l1_loss
from hanser.ops import to_float, to_int, all_reduce_mean
from hanser.detection.iou import bbox_iou2


def distribution_focal_loss(y_true, y_pred):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        y_true (tf.Tensor): Target distance label for bounding boxes with
            shape (N,).
        y_pred (tf.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = tf.cast(y_true, tf.int32)
    dis_right = dis_left + 1
    weight_left = tf.cast(dis_right, tf.float32) - y_true
    weight_right = y_true - tf.cast(dis_left, tf.float32)
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy
    loss = cross_entropy(dis_left, y_pred, from_logits=True) * weight_left \
        + cross_entropy(dis_right, y_pred, from_logits=True) * weight_right
    return loss


class GFLossV2:

    def __init__(self, bbox_coder, box_loss_weight=2.,
                 iou_loss_mode='giou', quantity_mode='iou'):
        self.bbox_coder = bbox_coder
        self.box_loss_weight = box_loss_weight
        self.iou_loss_mode = iou_loss_mode
        self.quantity_mode = quantity_mode

    def __call__(self, y_true, y_pred):

        bbox_targets = y_true['bbox_target']
        labels = y_true['label']

        dis_logits = y_pred['dis_logit']
        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']

        pos = labels != 0
        pos_weight = to_float(pos)
        total_pos = tf.reduce_sum(pos_weight)
        total_pos = all_reduce_mean(total_pos)

        reg_max = dis_logits.shape[-1] - 1
        bbox_targets = self.bbox_coder.encode(bbox_targets)
        scaled_targets = tf.clip_by_value(bbox_targets / y_pred['scales'][0], 0, reg_max - 0.001)
        loss_dfl = distribution_focal_loss(scaled_targets, dis_logits)

        quantity_scores = bbox_iou2(bbox_targets, bbox_preds,
                                    mode=self.quantity_mode, is_aligned=True, offset=True)
        quantity_scores = tf.stop_gradient(quantity_scores)
        loss_cls = quality_focal_loss(
            (labels, quantity_scores), cls_scores, from_logits=False, reduction='sum') / total_pos

        cls_scores = tf.reduce_max(tf.stop_gradient(cls_scores), axis=-1)
        weight = cls_scores * pos_weight
        avg_factor = tf.reduce_sum(weight)
        avg_factor = all_reduce_mean(avg_factor)
        loss_box = iou_loss(bbox_targets, bbox_preds, weight=weight,
                            reduction='sum', mode=self.iou_loss_mode, offset=True) / avg_factor

        loss_dfl = tf.reduce_sum(loss_dfl * weight[:, :, None]) / (avg_factor * 4)
        loss = loss_box * self.box_loss_weight + loss_cls + loss_dfl
        return loss


class GFLoss:

    def __init__(self, bbox_coder, decode_pred=True, box_loss_weight=2.,
                 iou_loss_mode='giou', quantity_mode='iou', offset=False,
                 quantity_weighted=True):
        self.bbox_coder = bbox_coder
        self.decode_pred = decode_pred
        self.box_loss_weight = box_loss_weight
        self.iou_loss_mode = iou_loss_mode
        self.quantity_mode = quantity_mode
        self.offset = offset
        self.quantity_weighted = quantity_weighted

    def __call__(self, y_true, y_pred):

        bbox_targets = y_true['bbox_target']
        labels = y_true['label']

        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']

        pos = labels != 0
        pos_weight = to_float(pos)
        total_pos = tf.reduce_sum(pos_weight)
        total_pos = all_reduce_mean(total_pos)

        if self.decode_pred:
            bbox_preds = self.bbox_coder.decode(bbox_preds)

        quantity_scores =  bbox_iou2(bbox_targets, bbox_preds,
                                     mode=self.quantity_mode, is_aligned=True, offset=self.offset)
        quantity_scores = tf.stop_gradient(quantity_scores)
        loss_cls = quality_focal_loss(
            (labels, quantity_scores), cls_scores, reduction='sum') / total_pos

        if self.quantity_weighted:
            # TODO: May be better to use groundtruth
            cls_scores = tf.reduce_max(tf.stop_gradient(cls_scores), axis=-1)
            cls_scores = tf.sigmoid(cls_scores)
            box_losses_weight = cls_scores * pos_weight
            box_loss_avg_factor = tf.reduce_sum(box_losses_weight)
            box_loss_avg_factor = all_reduce_mean(box_loss_avg_factor)
        else:
            box_losses_weight = pos_weight
            box_loss_avg_factor = total_pos
        loss_box = iou_loss(bbox_targets, bbox_preds, weight=box_losses_weight,
                            reduction='sum', mode=self.iou_loss_mode, offset=self.offset) / box_loss_avg_factor

        loss = loss_box * self.box_loss_weight + loss_cls
        return loss


class DetectionLoss:

    def __init__(self, box_loss_fn, cls_loss_fn, box_loss_weight=1.,
                 bbox_coder=None, decode_pred=False, centerness=False,
                 quantity_weighted=True):
        if decode_pred:
            assert bbox_coder is not None
        self.box_loss_fn = box_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.box_loss_weight = box_loss_weight
        self.bbox_coder = bbox_coder
        self.decode_pred = decode_pred
        self.centerness = centerness
        self.quantity_weighted = quantity_weighted

    def __call__(self, y_true, y_pred):
        bbox_targets = y_true['bbox_target']
        labels = y_true['label']
        ignore = y_true.get("ignore")
        if ignore is None:
            non_ignore = None
        else:
            non_ignore = to_float(~ignore)

        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']

        pos = labels != 0
        pos_weight = to_float(pos)
        total_pos = tf.reduce_sum(pos_weight)
        total_pos = all_reduce_mean(total_pos)

        if self.decode_pred:
            bbox_preds = self.bbox_coder.decode(bbox_preds)

        loss_cls = self.cls_loss_fn(labels, cls_scores, weight=non_ignore, reduction='sum') / total_pos
        loss = loss_cls

        if self.centerness:
            centerness = y_pred['centerness']
            centerness_t = y_true['centerness']
            loss_centerness = tf.nn.sigmoid_cross_entropy_with_logits(centerness_t, centerness)
            loss_centerness = reduce_loss(loss_centerness, pos_weight, reduction='sum') / total_pos
            loss = loss + loss_centerness

        box_losses_weight = pos_weight
        box_loss_avg_factor = total_pos
        if self.quantity_weighted and self.centerness:
            centerness_t = y_true['centerness']
            box_losses_weight = centerness_t
            box_loss_avg_factor = tf.reduce_sum(centerness_t)
            box_loss_avg_factor = all_reduce_mean(box_loss_avg_factor)

        loss_box = self.box_loss_fn(bbox_targets, bbox_preds,
                                    weight=box_losses_weight, reduction='sum') / box_loss_avg_factor

        loss = loss + loss_box * self.box_loss_weight
        return loss


@curry
def iou_loss(y_true, y_pred, weight=None, mode='iou', offset=False, reduction='sum'):
    # y_true: (batch_size, n_dts, 4)
    # y_pred: (batch_size, n_dts, 4)
    # weight: (batch_size, n_dts)
    losses = 1.0 - bbox_iou2(y_true, y_pred, mode=mode, is_aligned=True, offset=offset)
    return reduce_loss(losses, weight, reduction)


@curry
def cross_entropy_det(y_true, y_pred, weight=None, neg_pos_ratio=None, reduction='sum'):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    if neg_pos_ratio is None:
        return reduce_loss(losses, weight, reduction)

    assert reduction == 'sum'
    losses = losses * weight

    pos = tf.cast(y_true != 0, y_pred.dtype)
    loss_pos = reduce_loss(losses, pos, reduction)

    n_pos = tf.reduce_sum(pos, axis=1)
    neg = 1. - pos
    loss_neg = hard_negative_mining(losses * neg, n_pos, neg_pos_ratio)
    return loss_pos + loss_neg


def hard_negative_mining(losses, n_pos, neg_pos_ratio, max_pos=1000):
    ind = tf.range(max_pos, dtype=tf.int32)[None, :]
    n_neg = to_int(n_pos * neg_pos_ratio)
    weights = tf.cast(ind < n_neg[:, None], tf.float32)
    losses = tf.math.top_k(losses, k=max_pos, sorted=True)[0]
    return tf.reduce_sum(weights * losses)


def quality_focal_loss(y_true, y_pred, gamma=2.0, from_logits=True, reduction='sum'):
    label, score = y_true
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.one_hot(label, num_classes + 1)[..., 1:] * score[..., None]

    if from_logits:
        sigma = tf.sigmoid(y_pred)
        focal_weight = tf.abs(y_true - sigma) ** gamma
        losses = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    else:
        focal_weight = tf.abs(y_true - y_pred) ** gamma
        losses = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)
    losses = losses * focal_weight
    return reduce_loss(losses, weight=None, reduction=reduction)