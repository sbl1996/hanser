from toolz import curry

import tensorflow as tf
import tensorflow.keras.backend as K
from hanser.train.losses import CrossEntropy


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


def focal_loss(y_true, y_pred, alpha, gamma, label_smoothing=0.0, eps=1e-6):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        y_true: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        y_pred: A float tensor of arbitrary shape.
                The predictions for each example.
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = tf.sigmoid(y_pred)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    weight = (1 - p_t) ** gamma
    if alpha:
        weight = weight * (alpha * y_true + (1 - alpha) * (1 - y_true))
    if label_smoothing:
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_pred_ls = (1. - label_smoothing) * y_pred + label_smoothing / num_classes
        y_pred = tf.clip_by_value(y_pred_ls, eps, 1. - eps)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    loss = ce_loss * weight
    return loss