from toolz import curry

import tensorflow as tf
import tensorflow.keras.backend as K
from hanser.train.losses import CrossEntropy


@curry
def cross_entropy(y_true, y_pred, ignore_label=None):
    batch_size = tf.shape(y_true)[0]
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.reshape(y_true, [batch_size, -1])
    y_pred = tf.reshape(y_pred, [batch_size, -1])
    if ignore_label is not None:
        mask = tf.not_equal(y_true, ignore_label)
        weights = tf.cast(mask, y_pred.dtype)
        y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        num_valid = tf.reduce_sum(weights, axis=1)
        losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

        losses = tf.reduce_sum(losses * weights, axis=1)
        num_valid = tf.maximum(num_valid, 1.)
        losses = losses / num_valid
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


def focal_loss(targets, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
    logits: A float32 tensor of size [batch, height_in, width_in,
      num_predictions].
    targets: A float32 tensor of size [batch, height_in, width_in,
      num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.

    Returns:
    loss: A float32 scalar representing normalized total loss.
    """
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    total_loss = tf.reduce_sum(weighted_loss)
    return total_loss
