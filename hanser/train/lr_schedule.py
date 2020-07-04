import math

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class CosineDecayRestarts(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts."""

    def __init__(
            self,
            initial_learning_rate,
            t_0,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
            warmup=0,
            warmup_alpha=0,
            steps_per_epoch=None,
            name=None):
        """Applies cosine decay with restarts to the learning rate.

        See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
        with Warm Restarts. https://arxiv.org/abs/1608.03983

        When training a models, it is often recommended to lower the learning rate as
        the training progresses. This schedule applies a cosine decay function with
        restarts to an optimizer step, given a provided initial learning rate.
        It requires a `step` value to compute the decayed learning rate. You can
        just pass a TensorFlow variable that you increment at each training step.

        The schedule a 1-arg callable that produces a decayed learning
        rate when passed the current optimizer step. This can be useful for changing
        the learning rate value across different invocations of optimizer functions.

        The learning rate multiplier first decays
        from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
        restart is performed. Each new warm restart runs for `t_mul` times more
        steps and with `m_mul` times smaller initial learning rate.

        Example usage:
        ```python
        first_decay_steps = 1000
        lr_decayed_fn = (
          tf.keras.experimental.CosineDecayRestarts(
              initial_learning_rate,
              first_decay_steps))
        ```

        You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
        as the learning rate. The learning rate schedule is also serializable and
        deserializable using `tf.keras.optimizers.schedules.serialize` and
        `tf.keras.optimizers.schedules.deserialize`.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
            number. The initial learning rate.
          first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
          t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period
          m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period:
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
        Returns:
          A 1-arg callable learning rate schedule that takes the current optimizer
          step and outputs the decayed learning rate, a scalar `Tensor` of the same
          type as `initial_learning_rate`.
        """
        super(CosineDecayRestarts, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.t_0 = t_0
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.warmup = warmup
        self.warmup_alpha = warmup_alpha
        self.steps_per_epoch = steps_per_epoch
        self.name = name

        self.first_decay_steps = t_0
        self.warmup_steps = warmup
        if steps_per_epoch:
            self.first_decay_steps *= steps_per_epoch
            self.warmup_steps *= steps_per_epoch

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            warmup_alpha = tf.cast(self.warmup_alpha, dtype)

            def warmup(step):
                completed_fraction = step / warmup_steps
                decayed = (1 - warmup_alpha) * completed_fraction + warmup_alpha
                return decayed

            def cosine_decay(step):
                completed_fraction = step / first_decay_steps

                def compute_step(completed_fraction, geometric=False):
                    """Helper for `cond` operation."""
                    if geometric:
                        i_restart = tf.floor(
                            tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                            tf.math.log(t_mul))

                        sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                        completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

                    else:
                        i_restart = tf.floor(completed_fraction)
                        completed_fraction -= i_restart

                    return i_restart, completed_fraction

                i_restart, completed_fraction = tf.cond(
                    tf.equal(t_mul, 1.0),
                    lambda: compute_step(completed_fraction, geometric=False),
                    lambda: compute_step(completed_fraction, geometric=True))

                m_fac = m_mul ** i_restart
                cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
                    tf.constant(math.pi) * completed_fraction))
                decayed = (1 - alpha) * cosine_decayed + alpha
                return decayed

            global_step_recomp = tf.cast(step, dtype)
            decayed = tf.cond(
                tf.less(global_step_recomp, warmup_steps),
                lambda: warmup(global_step_recomp),
                lambda: cosine_decay(global_step_recomp - warmup_steps),
            )
            return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "t_0": self.t_0,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "warmup": self.warmup,
            "warmup_alpha": self.warmup_alpha,
            "steps_per_epoch": self.steps_per_epoch,
            "name": self.name,
            "first_decay_steps": self.first_decay_steps,
            "warmup_steps": self.warmup_steps,
        }



class CosineLR(LearningRateSchedule):

    def __init__(self,
            learning_rate,
            steps_per_epoch,
            epochs,
            min_lr,
            warmup_min_lr,
            warmup_epoch,
            name=None):

        super().__init__()

        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.warmup_min_lr = warmup_min_lr
        self.warmup_epoch = warmup_epoch
        self.name = name

        self.first_decay_steps = t_0
        self.warmup_steps = warmup
        if steps_per_epoch:
            self.first_decay_steps *= steps_per_epoch
            self.warmup_steps *= steps_per_epoch

    def __call__(self, step):
        initial_learning_rate = tf.convert_to_tensor(
            self.initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        first_decay_steps = tf.cast(self.first_decay_steps, dtype)
        alpha = tf.cast(self.alpha, dtype)
        t_mul = tf.cast(self.min_lr, dtype)
        m_mul = tf.cast(self._m_mul, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        warmup_alpha = tf.cast(self.warmup_alpha, dtype)

        def warmup(step):
            completed_fraction = step / warmup_steps
            decayed = (1 - warmup_alpha) * completed_fraction + warmup_alpha
            return decayed

        def cosine_decay(step):
            completed_fraction = step / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                        tf.math.log(t_mul))

                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul ** i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
                tf.constant(math.pi) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha
            return decayed

        global_step_recomp = tf.cast(step, dtype)
        decayed = tf.cond(
            tf.less(global_step_recomp, warmup_steps),
            lambda: warmup(global_step_recomp),
            lambda: cosine_decay(global_step_recomp - warmup_steps),
        )
        return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "t_0": self.t_0,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "warmup": self.warmup,
            "warmup_alpha": self.warmup_alpha,
            "steps_per_epoch": self.steps_per_epoch,
            "name": self.name,
            "first_decay_steps": self.first_decay_steps,
            "warmup_steps": self.warmup_steps,
        }