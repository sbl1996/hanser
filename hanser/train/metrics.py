from hanser.detection.eval import average_precision
from toolz import curry, get

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric, Mean


from hanser.detection import BBox


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
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    def __init__(self, num_classes, ignore_index, name=None, dtype=None):
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

    def update_state(self, y_true, y_pred, sample_weight=None):
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
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

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
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes), dtype=np.int32))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseCategoricalAccuracy(Mean):

    def __init__(self, ignore_index, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._ignore_index = ignore_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, self._dtype)
        # y_pred = tf.cast(y_pred, self._dtype)
        # [y_true, y_pred], sample_weight = \
        #     metrics_utils.ragged_assert_compatible_and_get_flat_values(
        #         [y_true, y_pred], sample_weight)
        # y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
        #     y_pred, y_true)
        #
        # matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        # return super(MeanMetricWrapper, self).update_state(
        #     matches, sample_weight=sample_weight)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)

        matches = tf.cast(tf.equal(y_true, y_pred), K.floatx())
        sample_weight = tf.not_equal(y_true, self._ignore_index)

        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {
            'ignore_index': self._ignore_index,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@curry
def sparse_categorical_accuracy(y_true, y_pred, ignore_label):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)

    n_correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    n_total = tf.reduce_sum(tf.cast(tf.not_equal(y_true, ignore_label), tf.float32))

    return n_correct / n_total


def confusion_matrix(y_true, y_pred, num_classes):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    c = num_classes
    return tf.reshape(tf.math.bincount(y_true * c + y_pred, minlength=c * c), (c, c))


@curry
def mean_iou(y_true, y_pred, num_classes, ignore_label=None, per_class=False):

    if y_pred.shape.ndims - y_true.shape.ndims == 1:
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
    # print(y_true.shape)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    if ignore_label is not None:
        mask = y_true != ignore_label
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    cm = confusion_matrix(y_true, y_pred, num_classes)
    # compute mean iou
    intersection = tf.linalg.diag_part(cm)
    ground_truth_set = tf.reduce_sum(cm, axis=1)
    predicted_set = tf.reduce_sum(cm, axis=0)

    union = ground_truth_set + predicted_set - intersection

    num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(union, 0), dtype=tf.float32))

    iou = tf.math.divide_no_nan(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32))
    if per_class:
        return iou
    else:
        return tf.math.divide_no_nan(tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)


class MeanAveragePrecision:
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
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    def __init__(self, iou_threshold=0.5, interpolation='11point', ignore_difficult=True, class_names=None, name=None, dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        self.name = name
        self.dtype = dtype
        self.iou_threshold = iou_threshold
        self.interpolation = interpolation
        self.ignore_difficult = ignore_difficult
        self.class_names = class_names

        self.gts = []
        self.dts = []

    def update_state(self, y_true, y_pred, sample_weight=None):
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
        all_dt_bboxes, all_dt_classes, all_dt_scores, all_dt_n_valids = [
            t.numpy()
            for t in get(['bbox', 'label', 'score', 'n_valid'], y_pred)
        ]

        all_gt_bboxes, all_gt_classes, all_is_difficults, image_ids = [
            t.numpy()
            for t in get(['bbox', 'label', 'is_difficult', 'image_id'], y_true)
        ]

        all_gt_n_valids = np.sum(all_gt_classes != 0, axis=1)
        all_gt_classes -= 1
        batch_size, num_dets = all_dt_bboxes.shape[:2]
        for i in range(batch_size):
            image_id = image_ids[i]
            n_valid = all_dt_n_valids[i]
            for j in range(n_valid):
                self.dts.append({
                    'image_id': image_id,
                    'bbox': all_dt_bboxes[i, j],
                    'category_id': all_dt_classes[i, j],
                    'score': all_dt_scores[i, j],
                })

            for j in range(all_gt_n_valids[i]):
                self.gts.append({
                    'image_id': image_id,
                    'bbox': all_gt_bboxes[i, j],
                    'category_id': all_gt_classes[i, j],
                    'is_difficult': all_is_difficults[i, j]
                })

    def result(self):
        dts = [BBox(**ann) for ann in self.dts]
        gts = [BBox(**ann) for ann in self.gts]

        aps = average_precision(dts, gts, self.iou_threshold, self.interpolation == '11point', self.ignore_difficult)
        mAP = np.mean(list(aps.values()))
        if self.class_names:
            num_classes = len(self.class_names)
            d = {}
            for i in range(num_classes):
                d[self.class_names[i]] = aps.get(i, 0) * 100
            d['ALL'] = mAP * 100
            d = pd.DataFrame({'mAP': d}).transpose()
            pd.set_option('precision', 1)
            print(d)
        return tf.convert_to_tensor(mAP)

    def reset_states(self):
        self.gts = []
        self.dts = []

    def get_config(self):
        config = {
            'iou_threshold': self.iou_threshold,
            'interpolation': self.interpolation,
            'ignore_difficult': self.ignore_difficult,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
