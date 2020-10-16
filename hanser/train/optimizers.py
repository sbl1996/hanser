from typing import Union, Callable

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Hanser")
class SGD(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        momentum: FloatTensorLike = 0.9,
        dampening: FloatTensorLike = 0.0,
        name: str = "SGD",
        **kwargs
    ):

        super().__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", momentum)
        self._set_hyper("dampening", dampening)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        momentum = tf.identity(self._get_hyper("momentum", var_dtype))
        dampening = tf.identity(self._get_hyper("dampening", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                momentum=momentum,
                one_minus_dampening=1 - dampening,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # v_t = mu * v + (1 - d) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m * coefficients["momentum"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)

        var_update = var - coefficients["lr_t"] * m_t
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m.assign(m * coefficients["momentum"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        var_update = var.assign_sub(
            coefficients["lr_t"] * m_t, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("beta_1"),
                "dampening": self._serialize_hyperparameter("beta_2"),
            }
        )
        return config