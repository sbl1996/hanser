import math

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops


class CosineAnnealingLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        min_lr,
        warmup_epoch=0,
        warmup_min_lr=0,
        epoch_annealing=False,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch
        self.epoch_annealing = epoch_annealing

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

    def __call__(self, step):

        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)

        step = tf.cast(step, dtype)
        if self.epoch_annealing:
            step2 = tf.floor(step / self.steps_per_epoch) * self.steps_per_epoch
        else:
            step2 = step

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def cosine_decay(step):
            frac = (step - warmup_steps) / (total_steps - warmup_steps)
            mult = (tf.cos(frac * tf.constant(math.pi)) + 1) / 2
            return min_lr + (base_lr - min_lr) * mult

        decayed_lr = tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: cosine_decay(step2),
        )
        return decayed_lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "min_lr": self.min_lr,
            "alpha": self.alpha,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "epoch_annealing": self.epoch_annealing,
        }


class FlatCosineLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        flat_epoch,
        min_lr,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.flat_epoch = flat_epoch
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch
        self.flat_steps = flat_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)
        flat_steps = tf.cast(self.flat_steps, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def cosine_decay(step):
            frac = (step - flat_steps) / (total_steps - flat_steps)
            mult = (tf.cos(frac * tf.constant(math.pi)) + 1) / 2
            return min_lr + (base_lr - min_lr) * mult

        global_step_recomp = tf.cast(step, dtype)
        decayed_lr = tf.cond(
            tf.less(global_step_recomp, warmup_steps),
            lambda: warmup(global_step_recomp),
            lambda: tf.cond(
                tf.less(global_step_recomp, flat_steps),
                lambda: base_lr,
                lambda: cosine_decay(global_step_recomp),
            ),
        )
        return decayed_lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "flat": self.flat,
            "min_lr": self.min_lr,
            "alpha": self.alpha,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


class CosinePowerAnnealingLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        p,
        min_lr,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        assert p > 1
        self.p = p
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)
        p = tf.cast(self.p, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def cosine_power_decay(step):
            frac = (step - warmup_steps) / (total_steps - warmup_steps)
            mult = (tf.cos(frac * tf.constant(math.pi)) + 1) / 2
            mult = (tf.pow(p, mult + 1) - p) / (tf.pow(p, 2) - p)
            return min_lr + (base_lr - min_lr) * mult

        global_step_recomp = tf.cast(step, dtype)
        decayed_lr = tf.cond(
            tf.less(global_step_recomp, warmup_steps),
            lambda: warmup(global_step_recomp),
            lambda: cosine_power_decay(global_step_recomp),
        )
        return decayed_lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "p": self.p,
            "min_lr": self.min_lr,
            "alpha": self.alpha,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


CosineLR = CosineAnnealingLR


class MultiStepLR(LearningRateSchedule):

    def __init__(self, learning_rate, steps_per_epoch, milestones, gamma, warmup_epoch=0, warmup_min_lr=0):
        super().__init__()
        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

        self.boundaries = [x * self.steps_per_epoch for x in self.milestones]
        self.values = [self.base_lr * (gamma ** i) for i in range(len(self.milestones) + 1)]

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
        boundaries = ops.convert_n_to_tensor(self.boundaries)
        values = ops.convert_n_to_tensor(self.values)

        dtype = base_lr.dtype
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)
        step = tf.cast(step, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        boundaries = [tf.cast(b, dtype) for b in boundaries]

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def step_decay(step):
            pred_fn_pairs = [
                (step <= boundaries[0], lambda: values[0]),
                (step > boundaries[-1], lambda: values[-1])
            ]
            for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
                pred = (step > low) & (step <= high)
                pred_fn_pairs.append((pred, lambda v=v: v))
            default = lambda: values[0]
            return tf.case(pred_fn_pairs, default, exclusive=True)

        decayed_lr = tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: step_decay(step - warmup_steps),
        )
        return decayed_lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "warmup_steps": self.warmup_steps,
            "boundaries": self.boundaries,
            "values": self.values,
        }


def scale_lr(lr, mul, mode='linear'):
    if mode == 'linear':
        return lr * mul
    elif mode == 'sqrt':
        return lr * math.sqrt(mul)
    else:
        raise ValueError("Not supported mode: %s" % mode)