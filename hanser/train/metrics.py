from hanser.detection import BBox
from hanser.detection.eval import average_precision
from toolz import get

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

    def __init__(self, num_classes, from_logits=True, name='miou', dtype=tf.int32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=Zeros(),
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        if self.from_logits:
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        current_cm = confusion_matrix(y_true, y_pred, self.num_classes, self.dtype)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        return iou_from_cm(self.total_cm)

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
                 ignore_label=None,
                 auxiliary_weight=0.0,
                 label_smoothing=0.0
                 ):
        super().__init__(cross_entropy, name, dtype=dtype,
                         ignore_label=ignore_label, auxiliary_weight=auxiliary_weight,
                         label_smoothing=label_smoothing)


class MeanAveragePrecision:

    def __init__(self, iou_threshold=0.5, interpolation='11point', ignore_difficult=True, class_names=None, output_transform=None):

        self.iou_threshold = iou_threshold
        self.interpolation = interpolation
        self.ignore_difficult = ignore_difficult
        self.class_names = class_names
        self.output_transform = output_transform

        self.gts = []
        self.dts = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.output_transform is not None:
            y_pred = self.output_transform(y_pred)

        all_dt_bboxes, all_dt_classes, all_dt_scores, all_dt_n_valids = [
            t.numpy() for t in get(['bbox', 'label', 'score', 'n_valid'], y_pred)
        ]

        all_gt_bboxes, all_gt_labels, all_is_difficults, image_ids = [
            t.numpy() for t in get(['bbox', 'label', 'is_difficult', 'image_id'], y_true)
        ]

        all_gt_n_valids = np.sum(all_gt_labels != 0, axis=1)
        all_gt_labels -= 1
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
                    'category_id': all_gt_labels[i, j],
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
