import tensorflow as tf


def cross_entropy(labels, logits, ignore_label=255):
    # labels = tf.cast(labels, tf.int32)

    num_classes = tf.shape(logits)[-1]

    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1, num_classes])

    mask = tf.cast(labels != ignore_label, logits.dtype)
    onehot_labels = tf.one_hot(labels, num_classes, off_value=0, on_value=1)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, mask)
    return loss
