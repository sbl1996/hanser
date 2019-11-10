from toolz import curry

import tensorflow as tf


@curry
def cross_entropy(labels, logits, ignore_label=None):
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
        labels = tf.where(mask, labels, tf.zeros_like(labels))
        weights = tf.cast(mask, logits.dtype)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, weights)
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return loss


@curry
def cross_entropy2(labels, logits, ignore_label=None):
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
        onehot_labels = tf.one_hot(labels, num_classes + 1)[..., :-1]
        loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits, from_logits=True)
        loss = tf.reduce_sum(loss) / num_valid
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return loss


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


def focal_loss(logits, targets, alpha, gamma):
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
