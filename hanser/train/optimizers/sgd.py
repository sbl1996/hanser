import re
from typing import Union, Callable, Optional, List, Tuple

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
        self.use_weight_decay = weight_decay != 0

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.use_weight_decay:
            return False
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
class SGDS(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable],
        momentum: FloatTensorLike = 0.9,
        dampening: FloatTensorLike = 0.0,
        weight_decay: FloatTensorLike = 0,
        nesterov: bool = False,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "SGDS",
        **kwargs
    ):
        super().__init__(name, **kwargs)

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
            bias_correction = coefficients["one_minus_dampening"] / (1 - coefficients["momentum"])
            weight_decay = bias_correction * coefficients["lr_t"] * coefficients["weight_decay"]
            var_update = var_update - weight_decay * var

        return var.assign(var_update, use_locking=self._use_locking)

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
class PNM(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable],
        betas: Tuple[FloatTensorLike, FloatTensorLike] = (0.9, 1.0),
        weight_decay: FloatTensorLike = 0,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "PNM",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta1", betas[0])
        self._set_hyper("beta2", betas[1])
        self._set_hyper("weight_decay", weight_decay)
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