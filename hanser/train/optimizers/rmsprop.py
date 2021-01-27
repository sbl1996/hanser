import re
import numpy as np
from typing import Optional, List

import tensorflow as tf
from tensorflow.python.training.gen_training_ops import resource_sparse_apply_centered_rms_prop, \
    resource_sparse_apply_rms_prop


@tf.keras.utils.register_keras_serializable(package="Hanser")
class RMSprop(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 learning_rate=0.01,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0.0,
                 centered=False,
                 exclude_from_weight_decay: Optional[List[str]] = None,
                 name="RMSprop",
                 **kwargs):

        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("alpha", alpha)

        self._momentum = False
        if tf.is_tensor(momentum) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)

        self.eps = eps
        self.centered = centered
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "square_avg")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "grad_avg")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        alpha = tf.identity(self._get_hyper("alpha", var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        momentum = tf.identity(self._get_hyper("momentum", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                eps=tf.convert_to_tensor(self.eps, var_dtype),
                alpha=alpha,
                momentum=momentum,
                one_minus_alpha=1. - alpha,
                weight_decay=weight_decay,
            ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        square_avg = self.get_slot(var, "square_avg")

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad = grad + coefficients["weight_decay"] * var

        square_avg_t = (coefficients["alpha"] * square_avg +
                 coefficients["one_minus_alpha"] * tf.math.square(grad))
        square_avg_t = square_avg.assign(square_avg_t, use_locking=self._use_locking)

        denom_t = square_avg_t
        if self.centered:
            grad_avg = self.get_slot(var, "grad_avg")
            grad_avg_t = coefficients["alpha"] * grad_avg + coefficients["one_minus_alpha"] * grad
            grad_avg_t = grad_avg.assign(grad_avg_t, use_locking=self._use_locking)
            denom_t = square_avg_t - tf.math.square(grad_avg_t)
        avg = tf.math.sqrt(denom_t) + coefficients["eps"]

        if self._momentum:
            momentum = self.get_slot(var, "momentum")
            momentum_t = momentum * coefficients['momentum'] + grad / avg
            update = momentum.assign(momentum_t, use_locking=self._use_locking)
        else:
            update = grad / avg
        var_t = var - coefficients["lr_t"] * update
        return var.assign(var_t, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = ((apply_state or {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      square_avg = self.get_slot(var, "square_avg")
      if self._momentum:
        mom = self.get_slot(var, "momentum")
        if self.centered:
          grad_avg = self.get_slot(var, "grad_avg")
          return resource_sparse_apply_centered_rms_prop(
              var.handle,
              grad_avg.handle,
              square_avg.handle,
              mom.handle,
              coefficients["lr_t"],
              coefficients["alpha"],
              coefficients["momentum"],
              coefficients["eps"],
              grad,
              indices,
              use_locking=self._use_locking)
        else:
          return resource_sparse_apply_rms_prop(
              var.handle,
              square_avg.handle,
              mom.handle,
              coefficients["lr_t"],
              coefficients["alpha"],
              coefficients["momentum"],
              coefficients["eps"],
              grad,
              indices,
              use_locking=self._use_locking)
      else:
        rms_scaled_g_values = (grad * grad) * coefficients["one_minus_alpha"]
        rms_t = square_avg.assign(square_avg * coefficients["alpha"],
                                  use_locking=self._use_locking)
        with tf.control_dependencies([rms_t]):
          rms_t = self._resource_scatter_add(square_avg, indices, rms_scaled_g_values)
          rms_slice = tf.gather(rms_t, indices)
        denom_slice = rms_slice
        if self.centered:
          grad_avg = self.get_slot(var, "grad_avg")
          grad_avg_scaled_g_values = grad * coefficients["one_minus_alpha"]
          grad_avg_t = grad_avg.assign(grad_avg * coefficients["alpha"],
                                       use_locking=self._use_locking)
          with tf.control_dependencies([grad_avg_t]):
            grad_avg_t = self._resource_scatter_add(grad_avg, indices, grad_avg_scaled_g_values)
            grad_avg_slice = tf.gather(grad_avg_t, indices)
            denom_slice = rms_slice - tf.math.square(grad_avg_slice)
        var_update = self._resource_scatter_add(
            var, indices, coefficients["neg_lr_t"] * grad / (
                tf.math.sqrt(denom_slice) + coefficients["eps"]))
        if self.centered:
          return tf.group(*[var_update, rms_t, grad_avg_t])
        return tf.group(*[var_update, rms_t])

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "alpha": self._serialize_hyperparameter("alpha"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "eps": self.eps,
            "centered": self.centered,
        })
        return config


RMSProp = RMSprop
