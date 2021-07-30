from hanser.train.metrics.common import MeanMetricWrapper
from hanser.losses import cross_entropy

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
