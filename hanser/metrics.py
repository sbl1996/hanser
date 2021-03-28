import tensorflow as tf


def confusion_matrix(y_true, y_pred, num_classes):
    class_indices = tf.range(num_classes)
    tm = tf.equal(y_true[:, None], class_indices[None, :])
    pm = tf.equal(y_pred[:, None], class_indices[None, :])
    cm = tf.logical_and(tm[:, :, None], pm[:, None, :])
    cm = tf.reduce_sum(tf.cast(cm, tf.int64), axis=0)
    return cm


def iou_from_cm(cm):
    intersection = tf.linalg.diag_part(cm)
    ground_truth_set = tf.reduce_sum(cm, axis=1)
    predicted_set = tf.reduce_sum(cm, axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union
    return IoU


def iou(y_true, y_pred, num_classes):
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return iou_from_cm(cm)


def mean_iou(y_true, y_pred, num_classes):
    return tf.reduce_mean(iou(y_true, y_pred, num_classes))
