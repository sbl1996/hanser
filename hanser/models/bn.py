import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization as KerasBatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization as KerasSyncBatchNormalization


class BatchNormalization(KerasBatchNormalization):

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
                 virtual_batch_size=None, adjustment=None, track_running_stats=True, eval_mode=False, name=None, **kwargs):
        super().__init__(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer, moving_mean_initializer,
                         moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint,
                         renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum, fused=fused, trainable=trainable,
                         virtual_batch_size=None, adjustment=adjustment, name=name, **kwargs)
        self._virtual_batch_size = virtual_batch_size
        self.track_running_stats = track_running_stats
        self.eval_mode = eval_mode

    def call(self, inputs, training=None):
        if self.eval_mode:
            return super().call(inputs, training=False)
        elif not self.track_running_stats:
            return super().call(inputs, training=True)
        elif training and self._virtual_batch_size is not None:
            batch_size = inputs.shape[0]
            q, r = divmod(batch_size, self._virtual_batch_size)
            num_or_size_splits = q
            if r != 0:
                num_or_size_splits = [self._virtual_batch_size] * q + [r]

            splits = tf.split(inputs, num_or_size_splits)
            x = [super(BatchNormalization, self).call(x, training=True) for x in splits]
            return tf.concat(x, 0)
        else:
            return super().call(inputs, training=training)


class SyncBatchNormalization(KerasSyncBatchNormalization):

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, trainable=True, adjustment=None,
                 track_running_stats=True, eval_mode=False, name=None, **kwargs):

        super().__init__(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer, moving_mean_initializer,
                         moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint,
                         renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum, trainable=trainable,
                         adjustment=adjustment, name=name, **kwargs)
        self.track_running_stats = track_running_stats
        self.eval_mode = eval_mode

    def call(self, inputs, training=None):
        if self.eval_mode:
            return super().call(inputs, training=False)
        elif not self.track_running_stats:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=training)
