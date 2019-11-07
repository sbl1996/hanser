import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU, Accuracy

from hanser.train.metrics import mean_iou, sparse_categorical_accuracy


def test_metrics():
    num_classes = 3
    ignore_label = 255

    def output_transform(output):
        return tf.argmax(output, axis=-1, output_type=tf.int32)

    def target_transform(target):
        return tf.where(tf.not_equal(target, ignore_label), target, tf.zeros_like(target))

    def get_sample_weight(labels, preds):
        weights = tf.not_equal(labels, ignore_label)
        return weights

    m1 = MeanIoU(num_classes, 'miou', dtype=tf.float32)
    m2 = Accuracy('acc', dtype=tf.float32)

    n = 100
    targets = []
    preds = []
    for i in range(n):
        labels = np.random.randint(0, num_classes, size=(2, 3, 3), dtype=np.int32)
        labels[np.random.normal(size=labels.shape) > 0] = ignore_label
        logits = np.random.normal(size=(2, 3, 3, num_classes)).astype(np.float32)

        labels = tf.convert_to_tensor(labels)
        logits = tf.convert_to_tensor(logits)

        targets.append(labels)
        preds.append(logits)

        weight = get_sample_weight(labels, logits)
        labels = target_transform(labels)
        pred = output_transform(logits)
        m1.update_state(labels, pred, weight)
        m2.update_state(labels, pred, weight)

    targets = tf.concat(targets, 0)
    preds = tf.concat(preds, 0)
    np.testing.assert_allclose(m1.result(), mean_iou(targets, preds, num_classes, ignore_label))
    np.testing.assert_allclose(m2.result(), sparse_categorical_accuracy(targets, preds, ignore_label))
