import math

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CosineAnnealingLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        min_lr=0,
        warmup_epoch=0,
        warmup_min_lr=0,
        staircase=False,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch
        self.staircase = staircase

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
        if self.staircase:
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
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "staircase": self.staircase,
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
            "flat_epoch": self.flat_epoch,
            "min_lr": self.min_lr,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "flat_steps": self.flat_steps,
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
        boundaries = [ tf.convert_to_tensor(b) for b in self.boundaries ]
        values = [ tf.convert_to_tensor(v) for v in self.values ]

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
            ]
            for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
                pred = (step > low) & (step <= high)
                pred_fn_pairs.append((pred, lambda v=v: v))
            pred_fn_pairs.append((step > boundaries[-1], lambda: values[-1]))
            default = lambda: values[0]
            return tf.case(pred_fn_pairs, default)

        decayed_lr = tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: step_decay(step),
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


class ExponentialDecay(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        decay_epochs,
        decay_rate,
        staircase=False,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):

        super().__init__()
        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.decay_steps = decay_epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
        dtype = base_lr.dtype
        decay_steps = tf.cast(self.decay_steps, dtype)
        decay_rate = tf.cast(self.decay_rate, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)

        step = tf.cast(step, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def exponential_decay(step):
            p = step / decay_steps
            if self.staircase:
                p = tf.math.floor(p)
            return tf.math.multiply(base_lr, tf.math.pow(decay_rate, p))

        return tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: exponential_decay(step - warmup_steps),
        )

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "decay_epochs": self.decay_epochs,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
        }


class ExponentialDecay2(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        decay_rate,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):

        super().__init__()
        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.decay_rate = decay_rate
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.warmup_steps = warmup_epoch * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
        dtype = base_lr.dtype
        decay_rate = tf.cast(self.decay_rate, dtype)
        total_steps = tf.cast(self.total_steps, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)

        step = tf.cast(step, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def exponential_decay(step):
            p = (step - warmup_steps) / (total_steps - warmup_steps)
            return tf.math.multiply(base_lr, tf.math.pow(decay_rate, p))

        return tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: exponential_decay(step),
        )

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "decay_rate": self.decay_rate,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
        }


class PolynomialDecay(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        power,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):
        super().__init__()
        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.power = power
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr, name="base_lr")
        dtype = base_lr.dtype
        power = tf.cast(self.power, dtype)
        total_steps = tf.cast(self.total_steps, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)

        step = tf.cast(step, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def poly_decay(step):
            p = (step - warmup_steps) / (total_steps - warmup_steps)
            return tf.math.multiply(base_lr, tf.math.pow(1 - p, power))

        return tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: poly_decay(step),
        )

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "power": self.power,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
        }


def scale_lr(lr, mul, mode='linear'):
    if mode == 'linear':
        return lr * mul
    elif mode == 'sqrt':
        return lr * math.sqrt(mul)
    else:
        raise ValueError("Not supported mode: %s" % mode)


class OneCycleLR(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        pct_start=0.3,
        anneal_strategy='linear',
        # cycle_momentum=True,
        # base_momentum=0.85,
        # max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):
        super().__init__()
        assert anneal_strategy in ['cos', 'linear']
        min_lr = learning_rate / div_factor
        self.max_lr = learning_rate
        self.min_lr = min_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.final_lr = learning_rate / final_div_factor
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = warmup_epoch * steps_per_epoch
        self.cycle_steps = int(self.total_steps * pct_start * 2)

        if self.warmup_epoch != 0:
            assert self.warmup_min_lr <= self.min_lr
            assert self.warmup_steps + self.cycle_steps < self.total_steps
        else:
            assert self.cycle_steps <= self.total_steps

    def __call__(self, step):
        max_lr = tf.convert_to_tensor(self.max_lr)
        dtype = max_lr.dtype
        min_lr = tf.cast(self.min_lr, dtype)
        final_lr = tf.cast(self.final_lr, dtype)
        total_steps = tf.cast(self.total_steps, dtype)
        cycle_steps = tf.cast(self.cycle_steps, dtype)
        final_steps = total_steps - cycle_steps
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)

        def warmup(step):
            return warmup_min_lr + (min_lr - warmup_min_lr) * step / warmup_steps

        def cosine_decay(step):
            frac = step / cycle_steps
            mult = (tf.cos(frac * tf.constant(math.pi)) + 1) / 2
            return min_lr + (max_lr - min_lr) * mult

        def linear_decay(step):
            frac = step / cycle_steps
            mult = 1 - tf.abs(2 * frac - 1)
            return min_lr + (max_lr - min_lr) * mult

        def final_decay(step):
            return min_lr - step / final_steps * (min_lr - final_lr)

        if self.anneal_strategy == 'cos':
            cycle_decay_fn = cosine_decay
        else:
            cycle_decay_fn = linear_decay

        step = tf.cast(step, dtype)
        decayed_lr = tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: tf.cond(
                tf.less(step, warmup_steps + cycle_steps),
                lambda: cycle_decay_fn(step - warmup_steps),
                lambda: final_decay(step - warmup_steps - cycle_steps),
            ),
        )
        return decayed_lr

    def get_config(self):
        return {
            "learning_rate": self.max_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "pct_start": self.pct_start,
            "anneal_strategy": self.anneal_strategy,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
        }


class Knee(LearningRateSchedule):

    def __init__(
        self,
        learning_rate,
        steps_per_epoch,
        epochs,
        explore_epoch,
        min_lr=0,
        warmup_epoch=0,
        warmup_min_lr=0,
    ):
        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.explore_epoch = explore_epoch
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch

        self.total_steps = epochs * steps_per_epoch
        self.warmup_steps = warmup_epoch * steps_per_epoch
        self.explore_steps = explore_epoch * steps_per_epoch

    def __call__(self, step):
        base_lr = tf.convert_to_tensor(self.base_lr)
        dtype = base_lr.dtype
        total_steps = tf.cast(self.total_steps, dtype)
        min_lr = tf.cast(self.min_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_min_lr = tf.cast(self.warmup_min_lr, dtype)
        explore_steps = tf.cast(self.explore_steps, dtype)

        def warmup(step):
            return warmup_min_lr + (base_lr - warmup_min_lr) * step / warmup_steps

        def linear_decay(step):
            before_steps = warmup_steps + explore_steps
            frac = 1 - (step - before_steps) / (total_steps - before_steps)
            return min_lr + (base_lr - min_lr) * frac

        step = tf.cast(step, dtype)
        decayed_lr = tf.cond(
            tf.less(step, warmup_steps),
            lambda: warmup(step),
            lambda: tf.cond(
                tf.less(step, explore_steps + warmup_steps),
                lambda: base_lr,
                lambda: linear_decay(step),
            ),
        )
        return decayed_lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "explore_epoch": self.explore_epoch,
            "min_lr": self.min_lr,
            "warmup_min_lr": self.warmup_min_lr,
            "warmup_epoch": self.warmup_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "flat_steps": self.flat_steps,
        }