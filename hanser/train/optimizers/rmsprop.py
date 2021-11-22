import re
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Hanser")
class RMSprop(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 learning_rate=0.001,
                 decay=0.9,
                 momentum=0.9,
                 epsilon=1e-3,
                 weight_decay=0.0,
                 exclude_from_weight_decay=None,
                 name="RMSprop"):
        """Construct a new RMSprop optimizer.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.001.
          decay: Discounting factor for the history/coming gradient. Defaults to 0.9.
          momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "RMSprop".

        """
        super().__init__(name)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", decay)

        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        assert self._momentum
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)

        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.use_weight_decay = weight_decay != 0

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _do_use_weight_decay(self, param_name):
        if not self.use_weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
        for var in var_list:
            self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        decay = tf.identity(self._get_hyper("decay", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                decay=decay,
                momentum=tf.identity(self._get_hyper("momentum", var_dtype)),
                weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype)),
            ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad = grad + coefficients["weight_decay"] * var

        return tf.raw_ops.ResourceApplyRMSProp(
            var=var.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["decay"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")

        return tf.raw_ops.ResourceSparseApplyRMSProp(
            var=var.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["decay"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            indices=indices,
            use_locking=self._use_locking)

    def get_config(self):
        config = super(RMSprop, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "epsilon": self.epsilon,
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
        })
        return config
