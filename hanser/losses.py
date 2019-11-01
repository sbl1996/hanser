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
        mask = tf.not_equal(labels, ignore_label)
        weights = tf.cast(mask, logits.dtype)
        labels = tf.where(mask, labels, tf.zeros_like(labels))
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, weights)
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    return loss