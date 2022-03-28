import numpy as np
import tensorflow as tf

def mlvl_concat(xs, reps, dtype=tf.float32):
    xs = np.array(xs)
    ndim = len(xs.shape)
    xs = np.concatenate([
        np.tile(xs[i][None], (n,) + (1,) * (ndim - 1))
        for i, n in enumerate(reps)
    ], axis=0)
    return tf.constant(xs, dtype)