# import re
# import numpy as np
# from typing import Optional, List
#
# import tensorflow as tf
# from tensorflow.python.training.gen_training_ops import resource_sparse_apply_centered_rms_prop, \
#     resource_sparse_apply_rms_prop, resource_apply_rms_prop, resource_apply_centered_rms_prop
#
# @tf.keras.utils.register_keras_serializable(package="Hanser")
# class RMSprop(tf.keras.optimizers.Optimizer):
#     _HAS_AGGREGATE_GRAD = True
#
#     def __init__(self,
#                  learning_rate=0.001,
#                  alpha=0.9,
#                  eps=1e-7,
#                  weight_decay=0,
#                  momentum=0.0,
#                  centered=False,
#                  exclude_from_weight_decay: Optional[List[str]] = None,
#                  name="RMSprop",
#                  **kwargs):
#
#         super().__init__(name, **kwargs)
#         self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
#         self._set_hyper("decay", self._initial_decay)
#         self._set_hyper("alpha", alpha)
#
#         self._momentum = False
#         if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
#             self._momentum = True
#         if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
#             raise ValueError("`momentum` must be between [0, 1].")
#         self._set_hyper("momentum", momentum)
#         self._set_hyper("weight_decay", weight_decay)
#
#         self.eps = eps
#         self.centered = centered
#         self.exclude_from_weight_decay = exclude_from_weight_decay
#
#     def _get_variable_name(self, param_name):
#         """Get the variable name from the tensor name."""
#         m = re.match("^(.*):\\d+$", param_name)
#         if m is not None:
#             param_name = m.group(1)
#         return param_name
#
#     def _do_use_weight_decay(self, param_name):
#         """Whether to use L2 weight decay for `param_name`."""
#         if self.exclude_from_weight_decay:
#             for r in self.exclude_from_weight_decay:
#                 if re.search(r, param_name) is not None:
#                     return False
#         return True
#
#     def _create_slots(self, var_list):
#         for var in var_list:
#             self.add_slot(var, "rms")
#         if self._momentum:
#             for var in var_list:
#                 self.add_slot(var, "momentum")
#         if self.centered:
#             for var in var_list:
#                 self.add_slot(var, "mg")
#
#     def _prepare_local(self, var_device, var_dtype, apply_state):
#         super()._prepare_local(var_device, var_dtype, apply_state)
#
#         alpha = tf.identity(self._get_hyper("alpha", var_dtype))
#         weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
#         apply_state[(var_device, var_dtype)].update(
#             dict(
#                 neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
#                 eps=tf.convert_to_tensor(self.eps, var_dtype),
#                 alpha=alpha,
#                 momentum=tf.identity(self._get_hyper("momentum", var_dtype)),
#                 one_minus_alpha=1. - alpha,
#                 weight_decay = weight_decay,
#         ))
#
#     def _resource_apply_dense(self, grad, var, apply_state=None):
#         var_device, var_dtype = var.device, var.dtype.base_dtype
#         coefficients = ((apply_state or {}).get((var_device, var_dtype))
#                         or self._fallback_apply_state(var_device, var_dtype))
#
#         var_name = self._get_variable_name(var.name)
#         if self._do_use_weight_decay(var_name):
#             grad = grad + coefficients["weight_decay"] * var
#
#         rms = self.get_slot(var, "rms")
#         if self._momentum:
#             mom = self.get_slot(var, "momentum")
#             if self.centered:
#                 mg = self.get_slot(var, "mg")
#                 return resource_apply_centered_rms_prop(
#                     var.handle,
#                     mg.handle,
#                     rms.handle,
#                     mom.handle,
#                     coefficients["lr_t"],
#                     coefficients["alpha"],
#                     coefficients["momentum"],
#                     coefficients["eps"],
#                     grad,
#                     use_locking=self._use_locking)
#             else:
#                 return resource_apply_rms_prop(
#                     var.handle,
#                     rms.handle,
#                     mom.handle,
#                     coefficients["lr_t"],
#                     coefficients["alpha"],
#                     coefficients["momentum"],
#                     coefficients["eps"],
#                     grad,
#                     use_locking=self._use_locking)
#         else:
#             rms_t = (coefficients["alpha"] * rms +
#                      coefficients["one_minus_alpha"] * tf.square(grad))
#             rms_t = rms.assign(rms_t, use_locking=self._use_locking)
#             denom_t = rms_t
#             if self.centered:
#                 mg = self.get_slot(var, "mg")
#                 mg_t = coefficients["alpha"] * mg + coefficients["one_minus_alpha"] * grad
#                 mg_t = mg.assign(mg_t, use_locking=self._use_locking)
#                 denom_t = rms_t - tf.square(mg_t)
#             var_t = var - coefficients["lr_t"] * grad / (
#                 tf.sqrt(denom_t) + coefficients["eps"])
#             return var.assign(var_t, use_locking=self._use_locking)
#
#     def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
#         var_device, var_dtype = var.device, var.dtype.base_dtype
#         coefficients = ((apply_state or {}).get((var_device, var_dtype))
#                         or self._fallback_apply_state(var_device, var_dtype))
#
#         var_name = self._get_variable_name(var.name)
#         if self._do_use_weight_decay(var_name):
#             grad = grad + coefficients["weight_decay"] * var
#
#         rms = self.get_slot(var, "rms")
#         if self._momentum:
#             mom = self.get_slot(var, "momentum")
#             if self.centered:
#                 mg = self.get_slot(var, "mg")
#                 return resource_sparse_apply_centered_rms_prop(
#                     var.handle,
#                     mg.handle,
#                     rms.handle,
#                     mom.handle,
#                     coefficients["lr_t"],
#                     coefficients["alpha"],
#                     coefficients["momentum"],
#                     coefficients["eps"],
#                     grad,
#                     indices,
#                     use_locking=self._use_locking)
#             else:
#                 return resource_sparse_apply_rms_prop(
#                     var.handle,
#                     rms.handle,
#                     mom.handle,
#                     coefficients["lr_t"],
#                     coefficients["alpha"],
#                     coefficients["momentum"],
#                     coefficients["eps"],
#                     grad,
#                     indices,
#                     use_locking=self._use_locking)
#         else:
#             rms_scaled_g_values = (grad * grad) * coefficients["one_minus_alpha"]
#             rms_t = rms.assign(rms * coefficients["alpha"],
#                                use_locking=self._use_locking)
#             with tf.control_dependencies([rms_t]):
#                 rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
#                 rms_slice = tf.gather(rms_t, indices)
#             denom_slice = rms_slice
#             if self.centered:
#                 mg = self.get_slot(var, "mg")
#                 mg_scaled_g_values = grad * coefficients["one_minus_alpha"]
#                 mg_t = mg.assign(mg * coefficients["alpha"],
#                                  use_locking=self._use_locking)
#                 with tf.control_dependencies([mg_t]):
#                     mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
#                     mg_slice = tf.gather(mg_t, indices)
#                     denom_slice = rms_slice - tf.square(mg_slice)
#             var_update = self._resource_scatter_add(
#                 var, indices, coefficients["neg_lr_t"] * grad / (
#                     tf.sqrt(denom_slice) + coefficients["eps"]))
#             if self.centered:
#                 return tf.group(*[var_update, rms_t, mg_t])
#             return tf.group(*[var_update, rms_t])
#
#     def set_weights(self, weights):
#         params = self.weights
#         # Override set_weights for backward compatibility of Keras V1 optimizer
#         # since it does not include iteration at head of the weight list. Set
#         # iteration to 0.
#         if len(params) == len(weights) + 1:
#             weights = [np.array(0)] + weights
#         super().set_weights(weights)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "learning_rate": self._serialize_hyperparameter("learning_rate"),
#             "decay": self._serialize_hyperparameter("decay"),
#             "alpha": self._serialize_hyperparameter("alpha"),
#             "momentum": self._serialize_hyperparameter("momentum"),
#             "eps": self.eps,
#             "weight_decay": self._serialize_hyperparameter("weight_decay"),
#             "centered": self.centered,
#         })
#         return config
#
# RMSProp = RMSprop