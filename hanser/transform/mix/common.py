import tensorflow as tf

import tensorflow_probability as tfp

from hanser.ops import beta_mc

def _get_lam(shape, alpha, uniform=False, mc=False):
    if uniform:
        lam = tf.random.uniform(shape)
    elif mc:
        lam = beta_mc(alpha, alpha, shape, mc_size=10000)
    else:
        lam = tfp.distributions.Beta(alpha, alpha).sample(shape)
    return lam