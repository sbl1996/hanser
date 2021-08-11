import tensorflow as tf

from hanser.train.metrics.common import MeanMetricWrapper
from hanser.losses import cross_entropy
from hanser.ops import in_top_k

class CrossEntropy(MeanMetricWrapper):

    def __init__(self,
                 name='cross_entropy',
                 dtype=None,
                 ignore_label=None,
                 auxiliary_weight=0.0,
                 label_smoothing=0.0
                 ):
        super().__init__(cross_entropy, name, dtype=dtype,
                         ignore_label=ignore_label, auxiliary_weight=auxiliary_weight,
                         label_smoothing=label_smoothing)


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    result = in_top_k(y_pred, tf.argmax(y_true, axis=-1), k)
    return tf.cast(result, y_pred.dtype)


class TopKCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
    super(TopKCategoricalAccuracy, self).__init__(
        top_k_categorical_accuracy, name, dtype=dtype, k=k)