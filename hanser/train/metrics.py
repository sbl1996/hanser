import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Mean, Metric
from tensorflow.keras.initializers import Zeros

from hanser.losses import cross_entropy
from hanser.metrics import confusion_matrix, iou_from_cm


class MeanMetricWrapper(Mean):

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._compiled_fn = tf.function(fn)
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):

        matches = self._compiled_fn(y_true, y_pred, **self._fn_kwargs)
        return super().update_state(matches, sample_weight=None)

    def get_config(self):
        config = {}

        if type(self) is MeanMetricWrapper:  # pylint: disable=unidiomatic-typecheck
            # Only include function argument when the object is a MeanMetricWrapper
            # and not a subclass.
            config['fn'] = self._fn

        for k, v in self._fn_kwargs.items():
            config[k] = K.eval(v) if tf.is_tensor(v) else v
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Note that while MeanMetricWrapper itself isn't public, objects of this
        # class may be created and added to the model by calling model.compile.
        fn = config.pop('fn', None)
        if cls is MeanMetricWrapper:
            return cls(tf.keras.metrics.get(fn), **config)
        return super(MeanMetricWrapper, cls).from_config(config)


class MeanIoU(Metric):

    def __init__(self, num_classes, from_logits=True, name='miou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=Zeros(),
            dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        if self.from_logits:
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        current_cm = confusion_matrix(y_true, y_pred, self.num_classes)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        total_cm = tf.cast(self.total_cm, tf.float32)
        total_cm = tf.cast(total_cm.numpy(), tf.int32)
        return tf.reduce_mean(iou_from_cm(total_cm))

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'from_logits': self.from_logits,
            **super().get_config(),
        }


class CrossEntropy(MeanMetricWrapper):

    def __init__(self,
                 name='cross_entropy',
                 dtype=None,
                 ignore_label=None):
        super().__init__(cross_entropy, name, dtype=dtype, ignore_label=ignore_label)