import re
from typing import Union, Callable, Optional, List

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
        weight_decay: FloatTensorLike = 0,
        nesterov: bool = False,
        exclude_from_weight_decay: Optional[List[str]] = None,
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
        self._set_hyper("weight_decay", weight_decay)
        self.nesterov = nesterov
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        momentum = tf.identity(self._get_hyper("momentum", var_dtype))
        dampening = tf.identity(self._get_hyper("dampening", var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                momentum=momentum,
                one_minus_dampening=1 - dampening,
                weight_decay=weight_decay,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad = grad + coefficients["weight_decay"] * var

        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m * coefficients["momentum"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)

        if self.nesterov:
            update = grad + coefficients["momentum"] * m_t
        else:
            update = m_t

        var_update = var - coefficients["lr_t"] * update
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad = grad + coefficients["weight_decay"] * var

        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m.assign(m * coefficients["momentum"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        update = m_t

        var_update = var.assign_sub(
            coefficients["lr_t"] * update, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t])

    def get_config(self):
        return {
            **super().get_config(),
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "dampening": self._serialize_hyperparameter("dampening"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
        }

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


@tf.keras.utils.register_keras_serializable(package="Hanser")
class SGDW(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable],
        momentum: FloatTensorLike = 0.9,
        dampening: FloatTensorLike = 0.0,
        base_lr: Optional[FloatTensorLike] = None,
        weight_decay: FloatTensorLike = 0,
        nesterov: bool = False,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "SGD",
        **kwargs
    ):

        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("base_lr", self._get_base_lr(learning_rate, base_lr))
        self._set_hyper("momentum", momentum)
        self._set_hyper("dampening", dampening)
        self._set_hyper("weight_decay", weight_decay)
        self.nesterov = nesterov
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _get_base_lr(
        self, learning_rate: Union[FloatTensorLike, Callable], base_lr: Optional[FloatTensorLike]):
        if base_lr is not None:
            return float(base_lr)
        if isinstance(learning_rate, Callable):
            learning_rate = learning_rate(0)
        return float(learning_rate)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _do_use_weight_decay(self, param_name):
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        base_lr = tf.identity(self._get_hyper("base_lr", var_dtype))
        momentum = tf.identity(self._get_hyper("momentum", var_dtype))
        dampening = tf.identity(self._get_hyper("dampening", var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                base_lr=base_lr,
                momentum=momentum,
                one_minus_dampening=1 - dampening,
                weight_decay=weight_decay,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")

        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m * coefficients["momentum"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)

        if self.nesterov:
            update = grad + coefficients["momentum"] * m_t
        else:
            update = m_t

        var_update = var - coefficients["lr_t"] * update

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            weight_decay = coefficients["lr_t"] / coefficients['base_lr'] * coefficients["weight_decay"]
            var_update = var_update - weight_decay * var

        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # TODO: not implemented
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        m = self.get_slot(var, "m")

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad = grad + coefficients["weight_decay"] * var

        m_scaled_g_values = grad * coefficients["one_minus_dampening"]
        m_t = m.assign(m * coefficients["momentum"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        update = m_t

        var_update = var.assign_sub(
            coefficients["lr_t"] * update, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t])

    def get_config(self):
        return {
            **super().get_config(),
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "dampening": self._serialize_hyperparameter("dampening"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
        }

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
