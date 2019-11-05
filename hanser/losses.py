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
    if ignore_label is not None:
        num_classes = logits.shape[-1]
        mask = tf.not_equal(labels, ignore_label)
        labels = tf.where(mask, labels, tf.fill(labels.shape, num_classes))
        weights = tf.cast(mask, logits.dtype)
        num_valid = tf.reduce_sum(weights)
        onehot_labels = tf.one_hot(labels, num_classes + 1)[..., :-1]
        loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits, from_logits=True)
        loss = tf.reduce_sum(loss) / num_valid
        # loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, weights)
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
    if ignore_label is not None:
        num_classes = tf.shape(logits)[-1]
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
        num_classes = tf.shape(logits)[-1]
        onehot_labels = tf.one_hot(labels, num_classes, dtype=logits.dtype)
        if gamma > 1:
            logits = (onehot_labels * (gamma - 1) + 1) * logits
        if beta > 0:
            logits = logits + onehot_labels * beta
        loss = tf.keras.losses.categorical_crossentropy(onehot_labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return loss
