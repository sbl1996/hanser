from toolz import curry

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback

class LearningRateBatchScheduler(Callback):
    """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.epochs = -1
        self.prev_lr = -1

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.epochs += 1

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.schedule(self.epochs, batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if lr != self.prev_lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            self.prev_lr = lr

@curry
def step_lr_warmup(
        current_epoch, current_batch,
        base_lr, schedule, steps_per_epoch):
    epoch = current_epoch + float(current_batch) / steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = schedule[0]
    if epoch < warmup_end_epoch:
        # Learning rate increases linearly per step.
        return (base_lr * warmup_lr_multiplier *
                epoch / warmup_end_epoch)
    learning_rate = base_lr
    for mult, start_epoch in schedule:
        if epoch >= start_epoch:
            learning_rate = base_lr * mult
        else:
            break
    return learning_rate


class WeightDecay(Callback):

    def __init__(self, weight_decay):
        super().__init__()
        self.weight_decay = weight_decay

    def on_train_batch_end(self, batch, logs=None):
        for x in self.model.trainable_variables:
            if 'batch_normalization' not in x.name and self.weight_decay != 0:
                tf.assign_sub(x, self.weight_decay * x)
