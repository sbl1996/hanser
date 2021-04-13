from toolz import curry
import tensorflow as tf

from hanser.losses import reduce_loss, focal_loss, l1_loss, smooth_l1_loss
from hanser.ops import to_float, to_int
from hanser.detection.iou import bbox_iou2


class DetectionLoss:

    def __init__(self, box_loss_fn, cls_loss_fn, box_loss_weight=1.,
                 bbox_coder=None, decode_pred=False, centerness=False):
        if decode_pred or centerness:
            assert bbox_coder is not None
        self.box_loss_fn = box_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.box_loss_weight = box_loss_weight
        self.bbox_coder = bbox_coder
        self.decode_pred = decode_pred
        self.centerness = centerness

    def __call__(self, y_true, y_pred):

        bbox_targets = y_true['bbox_target']
        labels = y_true['label']
        non_ignore = to_float(~y_true['ignore'])

        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']

        pos = labels != 0
        pos_weight = to_float(pos)
        total_pos = tf.reduce_sum(pos_weight) + 1

        if self.decode_pred:
            dec_bbox_preds = self.bbox_coder.decode(bbox_preds)
            bbox_preds = dec_bbox_preds

        loss_box = self.box_loss_fn(bbox_targets, bbox_preds, weight=pos_weight, reduction='sum') / total_pos
        loss_cls = self.cls_loss_fn(labels, cls_scores, weight=non_ignore, reduction='sum') / total_pos

        loss = loss_box * self.box_loss_weight + loss_cls
        if self.centerness:
            centerness = y_pred['centerness']
            centerness_t = y_true['centerness']
            loss_centerness = tf.nn.sigmoid_cross_entropy_with_logits(
                centerness_t, centerness)
            loss_centerness = reduce_loss(loss_centerness, pos_weight, reduction='sum') / total_pos
            loss = loss + loss_centerness
        return loss


@curry
def iou_loss(y_true, y_pred, weight=None, mode='iou', reduction='sum'):
    # y_true: (batch_size, n_dts, 4)
    # y_pred: (batch_size, n_dts, 4)
    # weight: (batch_size, n_dts)
    losses = 1.0 - bbox_iou2(y_true, y_pred, mode=mode, is_aligned=True)
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
