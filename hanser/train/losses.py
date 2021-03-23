from tensorflow.keras.losses import CategoricalCrossentropy


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
