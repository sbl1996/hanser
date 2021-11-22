import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy


class CrossEntropy:

    def __init__(self, label_smoothing=0.0, reduction='none', auxiliary_weight=0.0):
        self._criterion = CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing,
                                                  reduction=reduction)
        self._auxiliary_weight = auxiliary_weight

    def __call__(self, y_true, y_pred):
        if self._auxiliary_weight:
            y_pred, y_pred_aux = y_pred
            loss = self._criterion(y_true, y_pred) + self._auxiliary_weight * self._criterion(y_true, y_pred_aux)
        else:
            loss = self._criterion(y_true, y_pred)
        return loss


class BinaryCrossEntropy:

    def __init__(self, label_smoothing=0.0, target_thresh=None, thresh_after_smooth=True,
                 reduction='sum'):
        assert reduction in ['sum', 'mean']
        self.label_smoothing = label_smoothing
        self.target_thresh = target_thresh
        self.thresh_after_smooth = thresh_after_smooth
        self.reduction = reduction

    def __call__(self, y_true, y_pred):
        if self.target_thresh and not self.thresh_after_smooth:
            y_true = tf.cast(y_true > self.target_thresh, y_pred.dtype)
        if self.label_smoothing:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        if self.target_thresh and self.thresh_after_smooth:
            y_true = tf.cast(y_true > self.target_thresh, y_pred.dtype)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        if self.reduction == 'sum':
            loss = tf.reduce_sum(loss, axis=-1)
        elif self.reduction == 'mean':
            loss = tf.reduce_mean(loss, axis=-1)
        return loss
