from toolz import curry

import tensorflow as tf
import tensorflow.keras.backend as K
from hanser.train.losses import CrossEntropy
from hanser.ops import to_float, to_int


@curry
def cross_entropy(y_true, y_pred, ignore_label=None, auxiliary_weight=0.0, label_smoothing=0.0):
    if auxiliary_weight:
        y_pred, y_pred_aux = y_pred
        return cross_entropy(y_true, y_pred, ignore_label, 0.0, label_smoothing) + \
               auxiliary_weight * cross_entropy(y_true, y_pred_aux, ignore_label, 0.0, label_smoothing)

    batch_size = tf.shape(y_true)[0]
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.reshape(y_true, [batch_size, -1])
    y_pred = tf.reshape(y_pred, [batch_size, -1, num_classes])
    if ignore_label is not None:
        mask = tf.not_equal(y_true, ignore_label)
        weights = tf.cast(mask, y_pred.dtype)
        y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        num_valid = tf.reduce_sum(weights, axis=1)

        if label_smoothing:
            y_true = tf.one_hot(y_true, num_classes, dtype=y_pred.dtype)
            losses = tf.keras.losses.categorical_crossentropy(
                y_true, y_pred, from_logits=True, label_smoothing=label_smoothing)
        else:
            losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

        losses = tf.reduce_sum(losses * weights, axis=1)
        num_valid = tf.maximum(num_valid, 1.)
        losses = losses / num_valid
    else:
        if label_smoothing:
            y_true = tf.one_hot(y_true, num_classes, dtype=y_pred.dtype)
            losses = tf.keras.losses.categorical_crossentropy(
                y_true, y_pred, from_logits=True, label_smoothing=label_smoothing)
        else:
            losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        losses = tf.reduce_mean(losses, axis=1)
    return losses


@curry
def f1_loss(y_true, y_pred, eps=1e-8):
    tp = tf.reduce_sum(y_pred * y_true)
    fp = tf.reduce_sum((1 - y_pred) * y_true)
    fn = tf.reduce_sum(y_pred * (1 - y_true))

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    return 1 - tf.reduce_mean(f1)


@curry
def weighted_bce(y_true, y_pred, pos_weight, from_logits=True):
    losses = K.binary_crossentropy(y_true, y_pred, from_logits)
    weight = tf.where(tf.equal(y_true, 1), pos_weight, 1.)
    losses = losses * weight
    return tf.reduce_mean(losses)


@curry
def focal_loss2(labels, logits, gamma=2, beta=1, ignore_label=None):
    r"""
    Args:
        labels: (N, H, W)
        logits: (N, H, W, C)
    """
    labels = tf.cast(labels, tf.int32)
    num_classes = tf.shape(logits)[-1]
    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1, num_classes])
    if ignore_label is not None:
        mask = tf.not_equal(labels, ignore_label)
        labels = tf.where(mask, labels, tf.fill(tf.shape(labels), num_classes))
        weights = tf.cast(mask, logits.dtype)
        num_valid = tf.reduce_sum(weights)
        onehot_labels = tf.one_hot(labels, num_classes + 1, dtype=logits.dtype)[..., :-1]
        if gamma > 1:
            logits = (onehot_labels * (gamma - 1) + 1) * logits
        if beta > 0:
            logits = logits + onehot_labels * beta
        loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits, from_logits=True)
        loss = tf.reduce_sum(loss) / num_valid
    else:
        onehot_labels = tf.one_hot(labels, num_classes, dtype=logits.dtype)
        if gamma > 1:
            logits = (onehot_labels * (gamma - 1) + 1) * logits
        if beta > 0:
            logits = logits + onehot_labels * beta
        loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return loss


@curry
def focal_loss(y_true, y_pred, weight=None, alpha=0.25, gamma=2.0, label_smoothing=0.0, label_offset=1, reduction='sum'):
    if y_pred.shape.ndims - y_true.shape.ndims == 1:
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, num_classes + label_offset, dtype=tf.float32)[..., 1:]
    p = tf.sigmoid(y_pred)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    focal_weight = (1 - p_t) ** gamma
    if alpha:
        focal_weight = focal_weight * (alpha * y_true + (1 - alpha) * (1 - y_true))
    if label_smoothing:
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_true = (1. - label_smoothing) * y_true + label_smoothing / num_classes
    losses = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    losses = losses * focal_weight
    return reduce_loss(losses, weight, reduction)


def reduce_loss(losses, weight=None, reduction='sum'):
    if weight is not None:
        if losses.shape.ndims - weight.shape.ndims == 1:
            weight = weight[..., None]
        losses = losses * weight
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return tf.reduce_sum(losses)
    else:
        return ValueError("Not supported reduction: %s" % reduction)


@curry
def smooth_l1_loss(y_true, y_pred, weight=None, beta=1.0, reduction='sum'):
    diff = tf.math.abs(y_pred - y_true)
    losses = tf.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta
    )
    return reduce_loss(losses, weight, reduction)


@curry
def l1_loss(y_true, y_pred, weight=None, clip_value=10, reduction='sum'):
    losses = tf.math.abs(y_pred - y_true)
    losses = tf.clip_by_value(losses, 0, clip_value)
    return reduce_loss(losses, weight, reduction)