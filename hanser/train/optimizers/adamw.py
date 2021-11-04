import re
from typing import Optional, List, Union, Callable

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Hanser")
class AdamW(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-8,
        amsgrad: bool = False,
        weight_decay: FloatTensorLike = 0.01,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name='AdamW',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('weight_decay', weight_decay)
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _do_use_weight_decay(self, param_name):
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        weight_decay = tf.identity(self._get_hyper('weight_decay', var_dtype))
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
              (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
                weight_decay=weight_decay,
            ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            wd = coefficients["weight_decay"] * coefficients["lr_t"]
            var = var.assign_sub(wd * var, self._use_locking)

        if not self.amsgrad:
            return tf.raw_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=coefficients['lr_t'],
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return tf.raw_ops.ResourceApplyAdamWithAmsgrad(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                vhat=vhat.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=coefficients['lr_t'],
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Not implemented
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = tf.assign(m, m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = tf.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = tf.sqrt(v_t)
            var_update = tf.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = tf.maximum(v_hat, v_t)
            with tf.control_dependencies([v_hat_t]):
                v_hat_t = tf.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = tf.sqrt(v_hat_t)
            var_update = tf.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._initial_decay,
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
