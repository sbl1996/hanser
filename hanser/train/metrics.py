from toolz import curry, get

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric, Mean


from hanser.detection import BBox
from hanser.detection.eval import average_precision



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
            for j in range(all_dt_n_valids[i]):
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
            # d = pd.DataFrame({'mAP': d}).transpose()
            # pd.set_option('precision', 1)
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


def calculate_mAP(gt_boxes, gt_labels, gt_difficulties, dt_boxes, dt_labels, dt_scores, n_classes=21):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param dt_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param dt_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param dt_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param gt_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param gt_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param gt_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(dt_boxes) == len(dt_labels) == len(dt_scores) == len(gt_boxes) == len(gt_labels) == len(gt_difficulties)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    gt_images = []
    for i in tf.range(len(gt_labels)):
        gt_images.append(tf.fill([len(gt_labels[i])], i))
    gt_images = tf.concat(gt_images, 0)
    gt_boxes = tf.concat(gt_boxes, 0)  # (n_objects, 4)
    gt_labels = tf.concat(gt_labels, 0)  # (n_objects)
    gt_difficulties = tf.concat(gt_difficulties, 0)  # (n_objects)

    assert gt_images.shape[0] == gt_boxes.shape[0] == gt_labels.shape[0] == gt_difficulties.shape[0]

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    dt_images = []
    for i in tf.range(len(dt_labels)):
        dt_images.append(tf.fill([len(dt_labels[i])], i))
    dt_images = tf.concat(dt_images, 0)
    dt_boxes = tf.concat(dt_boxes, 0)  # (n_detections, 4)
    dt_labels = tf.concat(dt_labels, 0)  # (n_detections)
    dt_scores = tf.concat(dt_scores, 0)  # (n_detections)

    assert dt_images.shape[0] == dt_boxes.shape[0] == dt_labels.shape[0] == dt_scores.shape[0]

    # Calculate APs for each class (except background)
    average_precisions = tf.zeros([n_classes - 1])  # (n_classes - 1)
    for c in tf.range(1, n_classes):
        # Extract only objects with this class
        gt_class_images = gt_images[gt_labels == c]  # (n_class_objects)
        gt_class_boxes = gt_boxes[gt_labels == c]  # (n_class_objects, 4)
        gt_class_difficulties = gt_difficulties[gt_labels == c]  # (n_class_objects)
        n_easy_class_objects = tf.reduce_sum(1 - gt_class_difficulties)

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        gt_class_boxes_detected = tf.zeros(len(gt_class_images), dtype=tf.uint8)
        # Extract only detections with this class
        dt_class_images = dt_images[dt_labels == c]  # (n_class_detections)
        dt_class_boxes = dt_boxes[dt_labels == c]  # (n_class_detections, 4)
        dt_class_scores = dt_scores[dt_labels == c]  # (n_class_detections)
        n_class_detections = len(dt_class_boxes)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        sort_ind = tf.argsort(dt_class_scores, axis=0, direction='DESCENDING')  # (n_class_detections)
        dt_class_scores = tf.gather(dt_class_scores, sort_ind)
        dt_class_images = tf.gather(dt_class_images, sort_ind)  # (n_class_detections)
        dt_class_boxes = tf.gather(dt_class_boxes, sort_ind)  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = tf.zeros([n_class_detections])  # (n_class_detections)
        false_positives = tf.zeros([n_class_detections])  # (n_class_detections)
        for d in tf.range(n_class_detections):
            this_detection_box = dt_class_boxes[d][None]  # (1, 4)
            this_image = dt_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = gt_class_boxes[gt_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = gt_class_difficulties[gt_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if len(object_boxes) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(gt_class_boxes.size(0)))[gt_class_images == this_image][ind]
            # We need 'original_ind' to update 'gt_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if gt_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        gt_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision
