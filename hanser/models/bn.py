from tensorflow.keras.layers import BatchNormalization as KerasBatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization as KerasSyncBatchNormalization


class BatchNormalization(KerasBatchNormalization):

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
                 virtual_batch_size=None, adjustment=None, track_running_stats=True, eval_mode=False, name=None, **kwargs):
        super().__init__(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer,
                         moving_mean_initializer, moving_variance_initializer, beta_regularizer, gamma_regularizer,
                         beta_constraint, gamma_constraint, renorm, renorm_clipping, renorm_momentum, fused, trainable,
                         virtual_batch_size, adjustment, name, **kwargs)
        if track_running_stats: assert not eval_mode
        if eval_mode: assert not track_running_stats

        self.track_running_stats = track_running_stats
        self.eval_mode = eval_mode

    def call(self, inputs, training=None):
        if self.eval_mode:
            return super().call(inputs, training=False)
        elif not self.track_running_stats:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=training)


class SyncBatchNormalization(KerasSyncBatchNormalization):

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, trainable=True, adjustment=None,
                 track_running_stats=True, eval_mode=False, name=None, **kwargs):

        super().__init__(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer,
                         moving_mean_initializer, moving_variance_initializer, beta_regularizer, gamma_regularizer,
                         beta_constraint, gamma_constraint, renorm, renorm_clipping, renorm_momentum, trainable,
                         adjustment, name, **kwargs)
        if track_running_stats: assert not eval_mode
        if eval_mode: assert not track_running_stats

        self.track_running_stats = track_running_stats
        self.eval_mode = eval_mode

    def call(self, inputs, training=None):
        if self.eval_mode:
            return super().call(inputs, training=False)
        elif not self.track_running_stats:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=training)
