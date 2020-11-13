import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.custom_gradient
def ShakeDropOp(x, p, alpha_min, alpha_max, beta_min, beta_max):
    gate = tf.random.uniform((), dtype=p.dtype) > (1 - p)
    alpha_min = tf.cast(alpha_min, x.dtype)
    alpha_max = tf.cast(alpha_max, x.dtype)
    shape = [tf.shape(x)[0], 1, 1, 1]
    out = tf.cond(
        gate,
        lambda: x * tf.random.uniform(shape, alpha_min, alpha_max, dtype=x.dtype),
        lambda: x)

    beta_min = tf.cast(beta_min, x.dtype)
    beta_max = tf.cast(beta_max, x.dtype)
    def custom_grad(dy):
        grad = tf.cond(
            gate,
            lambda: dy * tf.random.uniform(shape, beta_min, beta_max, dtype=dy.dtype),
            lambda: dy)
        return grad, None, None, None, None, None

    return out, custom_grad


class ShakeDrop(Layer):
    def __init__(self, p, alphas, betas, **kwargs):
        self.p = p
        self.alphas = alphas
        self.betas = betas
        super().__init__(**kwargs)

    def call(self, x, training):
        p = tf.cast(self.p, x.dtype)
        if training:
            return ShakeDropOp(x, p, self.alphas[0], self.alphas[1], self.betas[0], self.betas[1])
        else:
            return x * (1 - p)

    def get_config(self):
        base_config = super().get_config()
        base_config['p'] = self.p
        base_config['alphas'] = self.alphas
        base_config['betas'] = self.betas
        return base_config
