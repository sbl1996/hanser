import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from hanser.ops import confusion_matrix_tpu


def _iou_from_cm(cm):
    intersection = tf.linalg.diag_part(cm)
    ground_truth_set = tf.reduce_sum(cm, axis=1)
    predicted_set = tf.reduce_sum(cm, axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union

    mask = ground_truth_set != 0
    IoU = tf.where(mask, IoU, tf.zeros_like(IoU))
    return tf.reduce_sum(IoU) / tf.reduce_sum(tf.cast(mask, IoU.dtype))


def iou(y_true, y_pred, num_classes):
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    cm = confusion_matrix_tpu(y_true, y_pred, num_classes)
    return _iou_from_cm(cm)


def mean_iou(y_true, y_pred, num_classes):
    return tf.reduce_mean(iou(y_true, y_pred, num_classes))


class MeanIoU(Metric):

    def __init__(self, num_classes, from_logits=True, name='miou', dtype=tf.int32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        if self.from_logits:
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        current_cm = confusion_matrix_tpu(y_true, y_pred, self.num_classes, self.dtype)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        return _iou_from_cm(self.total_cm)

    def reset_states(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        return {
            **super().get_config(),
            'num_classes': self.num_classes,
            'from_logits': self.from_logits,
        }