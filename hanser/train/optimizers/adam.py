import re
from typing import Union, Callable, Optional, List, Tuple

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Hanser")
class AdaPNM(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable],
        betas: Tuple[FloatTensorLike, FloatTensorLike] = (0.9, 0.999, 1.0),
        eps: float = 1e-8,
        amsgrad: bool = True,
        weight_decay: FloatTensorLike = 0,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdaPNM",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta1", betas[0])
        self._set_hyper("beta2", betas[1])
        self._set_hyper("beta3", betas[2])
        self._set_hyper("weight_decay", weight_decay)

        self.eps = eps
        self.amsgrad = amsgrad
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pos_m")
            self.add_slot(var, "neg_m")

    def _do_use_weight_decay(self, param_name):
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        beta1 = tf.identity(self._get_hyper("beta1", var_dtype))
        beta2 = tf.identity(self._get_hyper("beta2", var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        is_even = tf.cast(self.iterations, tf.int32) % 2 == 0
        noise_norm = tf.sqrt((1 + beta2) ** 2 + beta2 ** 2)
        apply_state[(var_device, var_dtype)].update(
            dict(
                is_even=is_even,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                noise_norm=noise_norm,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        pos_m = self.get_slot(var, "pos_m")
        neg_m = self.get_slot(var, "neg_m")

        beta1, beta2 = coefficients['beta1'], coefficients['beta2']

        def momentum(m):
            return m * (beta1 ** 2) + grad * (1 - beta1 ** 2)

        is_even = coefficients['is_even']
        pos_m_t, neg_m_t = tf.cond(
            is_even,
            lambda: [momentum(pos_m), neg_m],
            lambda: [pos_m, momentum(neg_m)])

        pos_m = pos_m.assign(pos_m_t, use_locking=self._use_locking)
        neg_m = neg_m.assign(neg_m_t, use_locking=self._use_locking)

        pos_m_t, neg_m_t = tf.cond(is_even, lambda: [pos_m_t, neg_m_t], lambda: [neg_m_t, pos_m_t])
        update = ((1 + beta2) * pos_m_t - beta2 * neg_m_t) / coefficients['noise_norm']

        var_update = var - coefficients["lr_t"] * update

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            weight_decay = coefficients["lr_t"] * coefficients["weight_decay"]
            var_update = var_update - weight_decay * var

        return var.assign(var_update, use_locking=self._use_locking)

    def get_config(self):
        return {
            **super().get_config(),
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "betas": (self._serialize_hyperparameter("beta1"), self._serialize_hyperparameter("beta2")),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
        }

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name