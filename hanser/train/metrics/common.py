import tensorflow as tf
from tensorflow.keras.metrics import Mean


class MeanMetricWrapper(Mean):

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._compiled_fn = tf.function(lambda y_true, y_pred, **kwargs: fn(y_true, y_pred, **kwargs))
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):

        matches = self._compiled_fn(y_true, y_pred)
        return super().update_state(matches, sample_weight=None)

    def get_config(self):
        config = {}

        if type(self) is MeanMetricWrapper:  # pylint: disable=unidiomatic-typecheck
            # Only include function argument when the object is a MeanMetricWrapper
            # and not a subclass.
            config['fn'] = self._fn

        for k, v in self._fn_kwargs.items():
            config[k] = v.numpy() if tf.is_tensor(v) else v
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
