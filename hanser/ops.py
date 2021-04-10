import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1, return_index=False):
    gumbels = tfp.distributions.Gumbel(0, 1).sample(tf.shape(logits))
    gumbels = (logits + gumbels) / tau
    y_soft = tf.nn.softmax(gumbels, axis=axis)
    if hard:
        index = tf.argmax(y_soft, axis=axis, output_type=tf.int32)
        y_hard = tf.one_hot(index, tf.shape(logits)[-1], dtype=logits.dtype)
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
        if return_index:
            ret = ret, index
    else:
        ret = y_soft
    return ret


def nonzero(t):
    # assert t.ndim == 1 and t.dtype == tf.bool
    return tf.range(tf.shape(t)[0], dtype=tf.int32)[t]


def masked_scatter(t, mask, val):
    # assert t.dtype == val.dtype
    mask = tf.cast(mask, t.dtype)
    return tf.add(tf.multiply(t, 1 - mask), val * mask)


def index_put(t, indices, val):
    val = tf.cast(val, t.dtype)
    if val.shape.ndims == 0:
        val = tf.fill(tf.shape(indices), val)
    return tf.tensor_scatter_nd_update(t, indices[:, None], val)


def g(t, indices):
    return tf.gather(t, indices)


def to_float(x):
    return tf.cast(x, tf.float32)


def to_int(x):
    return tf.cast(x, tf.int32)


def choice(t, p=None):
    t = tf.convert_to_tensor(t)
    if p is None:
        p = tf.fill(t.shape, 1.0)
    p = to_float(p)[None]
    p = tf.math.log(p)
    i = tf.random.categorical(p, 1)[0, 0]
    return t[i]


def beta_mc(a, b, shape, mc_size=1000000):
    mc_table = tf.constant(np.random.beta(a, b, mc_size), dtype=tf.float32)
    indices = tf.random.uniform(shape, 0, mc_size, dtype=tf.int32)
    return tf.gather(mc_table, indices)


def misc_concat(values):
    if isinstance(values, (tuple, list)):
        val = values[0]
        if tf.is_tensor(val):
            return tf.concat(values, 0)
        elif isinstance(val, dict):
            d = {}
            for k in val.keys():
                d[k] = misc_concat([v[k] for v in values])
            return d
        elif isinstance(val, (tuple, list)):
            return val.__class__(v for l in values for v in l)
        else:
            return values
    elif isinstance(values, dict):
        return {k: misc_concat(v) for k, v in values.items()}
    else:
        return values


def get_shape(tensor, axis):
    shape = tensor.shape[axis]
    if shape is None:
        return tf.shape(tensor)[axis]
    else:
        return shape

def triu(x, diag=True):
    y = tf.linalg.band_part(x, 0, -1)
    if not diag:
        y = y - tf.linalg.band_part(x, 0, 0)
    return y
