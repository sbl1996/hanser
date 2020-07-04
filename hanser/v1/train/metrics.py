import tensorflow as tf

import numpy as np

from hanser.math import confusion_matrix
from tensorflow.keras.metrics import Metric


class MeanIoU(Metric):
    """Computes the mean Intersection-Over-Union metric.

    Mean Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation, which first computes the IOU for each semantic class and then
    computes the average over classes. IOU is defined as follows:
      IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state([0, 0, 1, 1], [0, 1, 0, 1])

      # cm = [[1, 1],
              [1, 1]]
      # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
      # iou = true_positives / (sum_row + sum_col - true_positives))
      # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    print('Final result: ', m.result().numpy())  # Final result: 0.33
    ```

    Usage with tf.keras API:

    ```python
    models = tf.keras.Model(inputs, outputs)
    models.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    def __init__(self, num_classes, ignore_index=None, name=None, dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.zeros_initializer,
            dtype=tf.int32)

    def update_state(self, y_true, y_pred):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        c = self.num_classes
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if self.ignore_index is not None:
            mask = tf.not_equal(y_true, self.ignore_index)
            y_pred = tf.where(mask, y_pred, tf.ones_like(y_pred) * c)
            y_true = tf.where(mask, y_true, tf.ones_like(y_true) * c)

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix(y_true, y_pred, self.num_classes + 1)[:c, :c]
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.div_no_nan(true_positives, denominator)

        return tf.div_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes), dtype=np.int32))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
