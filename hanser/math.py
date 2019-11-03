import tensorflow as tf


def confusion_matrix(y_true, y_pred, num_classes):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    c = num_classes
    return tf.reshape(tf.math.bincount(y_true * c + y_pred, minlength=c * c), (c, c))

