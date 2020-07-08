import numpy as np
from tensorflow.python.keras.metrics import MeanMetricWrapper
from hanser.losses import cross_entropy

def confusion_matrix(y_true, y_pred, num_classes):
    c = num_classes
    return np.bincount(y_true * c + y_pred, minlength=c * c).reshape(c, c)


def iou(y_true, y_pred, num_classes, ignore_index=None):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    if ignore_index is not None:
        mask = y_true != ignore_index
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    cm = confusion_matrix(y_true, y_pred, num_classes)
    # compute mean iou
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return IoU


def mean_iou(y_true, y_pred, num_classes, ignore_index=None):
    return np.mean(iou(y_true, y_pred, num_classes, ignore_index))


class CrossEntropy(MeanMetricWrapper):

  def __init__(self,
               name='cross_entropy',
               dtype=None,
               ignore_label=None,
               reduction='weighted_sum_by_nonzero_weights'):

    super().__init__(cross_entropy, name, dtype=dtype, ignore_label=ignore_label, reduction=reduction)