import re
from typing import Union, Callable, Optional, List

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked

@tf.keras.utils.register_keras_serializable(package="Hanser")
class RMSprop(tf.keras.optimizers.Optimizer):

  def __init__(self,
               learning_rate=0.01,
               alpha=0.99,
               eps=1e-8,
               weight_decay=0,
               momentum=0.0,
               centered=False,
               name="RMSprop",
               **kwargs):
    """Construct a new RMSprop optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defeaults to 0.001.
      rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
      momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSprop".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `decay`, `momentum`, and
    `epsilon` can each be a callable that takes no arguments and returns the
    actual value to use. This can be useful for changing these values across
    different invocations of optimizer functions.
    @end_compatibility
    """
    super(RMSprop, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("rho", rho)

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.epsilon = epsilon or backend_config.epsilon()
    self.centered = centered

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, "rms")
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")
    if self.centered:
      for var in var_list:
        self.add_slot(var, "mg")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(RMSprop, self)._prepare_local(var_device, var_dtype, apply_state)

    rho = array_ops.identity(self._get_hyper("rho", var_dtype))
    apply_state[(var_device, var_dtype)].update(
        dict(
            neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
            epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
            rho=rho,
            momentum=array_ops.identity(self._get_hyper("momentum", var_dtype)),
            one_minus_rho=1. - rho))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            use_locking=self._use_locking)
    else:
      rms_t = (coefficients["rho"] * rms +
               coefficients["one_minus_rho"] * math_ops.square(grad))
      rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
      denom_t = rms_t
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
        mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
        denom_t = rms_t - math_ops.square(mg_t)
      var_t = var - coefficients["lr_t"] * grad / (
          math_ops.sqrt(denom_t) + coefficients["epsilon"])
      return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_sparse_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            indices,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_sparse_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            indices,
            use_locking=self._use_locking)
    else:
      rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
      rms_t = state_ops.assign(rms, rms * coefficients["rho"],
                               use_locking=self._use_locking)
      with ops.control_dependencies([rms_t]):
        rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
        rms_slice = array_ops.gather(rms_t, indices)
      denom_slice = rms_slice
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_scaled_g_values = grad * coefficients["one_minus_rho"]
        mg_t = state_ops.assign(mg, mg * coefficients["rho"],
                                use_locking=self._use_locking)
        with ops.control_dependencies([mg_t]):
          mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
          mg_slice = array_ops.gather(mg_t, indices)
          denom_slice = rms_slice - math_ops.square(mg_slice)
      var_update = self._resource_scatter_add(
          var, indices, coefficients["neg_lr_t"] * grad / (
              math_ops.sqrt(denom_slice) + coefficients["epsilon"]))
      if self.centered:
        return control_flow_ops.group(*[var_update, rms_t, mg_t])
      return control_flow_ops.group(*[var_update, rms_t])

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(RMSprop, self).set_weights(weights)

  def get_config(self):
    config = super(RMSprop, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "rho": self._serialize_hyperparameter("rho"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "epsilon": self.epsilon,
        "centered": self.centered,
    })
    return config


RMSProp = RMSprop