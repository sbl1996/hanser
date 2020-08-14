import math

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CosineAnnealingLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        min_lr,
        warmup_min_lr,
        warmup_epoch,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
        warmup_min_lr = tf.convert_to_tensor(self.warmup_min_lr, name="warmup_min_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def cosine_decay(step):
            frac = (step - warmup_steps) / (total_steps - warmup_steps)
            mult = (tf.cos(frac * tf.constant(math.pi)) + 1) / 2
            return min_lr + (base_lr - min_lr) * mult

        global_step_recomp = tf.cast(step, dtype)
        decayed_lr = tf.cond(
            tf.less(global_step_recomp, warmup_steps),
            lambda: warmup(global_step_recomp),
            lambda: cosine_decay(global_step_recomp),
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
        }


class FlatCosineLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        flat_epoch,
        min_lr,
        warmup_min_lr,
        warmup_epoch,
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
        warmup_min_lr = tf.convert_to_tensor(self.warmup_min_lr, name="warmup_min_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
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
        warmup_min_lr,
        warmup_epoch,
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
        warmup_min_lr = tf.convert_to_tensor(self.warmup_min_lr, name="warmup_min_lr")

        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
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
