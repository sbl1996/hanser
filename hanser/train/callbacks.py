from toolz import curry

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class LearningRateBatchScheduler(Callback):
    """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

    def __init__(self, schedule, steps_per_epoch):
        super().__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.epochs = -1
        self.prev_lr = -1

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.epochs += 1

    def on_train_batch_begin(self, batch, logs=None):
        epochs = self.epochs + float(batch) / self.steps_per_epoch
        lr = self.schedule(epochs)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if lr != self.prev_lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            self.prev_lr = lr


@curry
def step_lr(epoch, base_lr, schedule):
    gamma, warmup = schedule[0]
    if epoch < warmup:
        eta_min = base_lr * gamma
        return eta_min + (base_lr - eta_min) * epoch / warmup
    learning_rate = base_lr
    for mult, start_epoch in schedule:
        if epoch >= start_epoch:
            learning_rate = base_lr * mult
        else:
            break
    return learning_rate


@curry
def cosine_lr(epoch, base_lr, total_epochs, warmup=0, gamma=0.1):
    if epoch < warmup:
        eta_min = base_lr * gamma
        return eta_min + (base_lr - eta_min) * epoch / warmup
    frac = (epoch - warmup) / (total_epochs - warmup)
    mult = (np.cos(frac * np.pi) + 1) / 2
    return base_lr * mult


@curry
def tf_cosine_lr(epoch, base_lr, total_epochs, warmup=0, gamma=0.1):
    eta_min = base_lr * gamma
    lr1 = eta_min + (base_lr - eta_min) * epoch / warmup

    frac = (epoch - warmup) / (total_epochs - warmup)
    lr2 = (tf.cos(frac * np.pi) + 1) / 2 * base_lr

    lr = tf.where(epoch < warmup, lr1, lr2)
    return lr


@curry
def one_cycle_lr(epoch, base_lr, max_lr, step_size, end_steps, gamma, warmup=0, warmup_gamma=0.1):
    if epoch < warmup:
        eta_min = base_lr * warmup_gamma
        lr = eta_min + epoch / warmup * (base_lr - eta_min)
    elif epoch < step_size + warmup:
        epoch -= warmup
        lr = base_lr + epoch / step_size * (max_lr - base_lr)
    elif epoch < (warmup + 2 * step_size):
        epoch -= warmup + step_size
        lr = base_lr + (step_size - epoch) / step_size * (max_lr - base_lr)
    elif epoch < (warmup + 2 * step_size + end_steps):
        epoch -= warmup + 2 * step_size
        lr = base_lr * gamma + epoch / end_steps * (base_lr - base_lr * gamma)
    else:
        lr = base_lr * gamma
    return lr
