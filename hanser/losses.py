from toolz import curry

import tensorflow as tf

@curry
def cross_entropy(labels, logits, ignore_label=None):
    r"""
    Args:
        labels: (N, H, W)
        logits: (N, H, W, C)
    """
    if ignore_label is not None:
        mask = tf.not_equal(labels, ignore_label)
        weights = tf.cast(mask, logits.dtype)
        labels = tf.where(mask, labels, tf.zeros_like(labels))
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) * weights
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss