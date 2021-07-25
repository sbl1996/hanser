from itertools import repeat
from typing import Tuple, Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1, return_index=False):
    u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    gumbels = -tf.math.log(-tf.math.log(u))
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


def beta_mc(a, b, shape, mc_size=10000):
    mc_table = tf.constant(np.random.beta(a, b, mc_size), dtype=tf.float32)
    indices = tf.random.uniform(shape, 0, mc_size, dtype=tf.int32)
    return tf.gather(mc_table, indices)


def log_uniform(shape, minval, maxval, dtype=tf.float32):
    minval = tf.math.log(minval)
    maxval = tf.math.log(maxval)
    x = tf.random.uniform(shape, minval, maxval, dtype)
    return tf.exp(x)


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


def l2_norm(x, sqrt=False):
    if sqrt:
        return tf.norm(x, axis=-1)
    return tf.reduce_sum(tf.square(x), axis=-1)


def _pair(x) -> Tuple:
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))


def _meshgrid(x, y, row_major=False):
    xx, yy = tf.meshgrid(x, y, indexing='xy' if row_major else 'ij')
    xx, yy = tf.reshape(xx, (-1,)), tf.reshape(yy, (-1,))
    return xx, yy


def all_reduce(tensor, op):
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None:
        return tensor
    return replica_context.all_reduce(op, tensor)

def all_reduce_mean(tensor):
    return all_reduce(tensor, tf.distribute.ReduceOp.MEAN)

def all_reduce_sum(tensor):
    return all_reduce(tensor, tf.distribute.ReduceOp.SUM)


def safe_softmax(logits, axis):
    dtype = logits.dtype
    if dtype in [tf.float16, tf.bfloat16]:
        logits = tf.cast(logits, tf.float32)
        weights = tf.nn.softmax(logits, axis=axis)
        weights = tf.cast(weights, dtype)
    else:
        weights = tf.nn.softmax(logits, axis=axis)
    return weights


def prepend_dims(x, n):
    for i in range(n):
        x = tf.expand_dims(x, 0)
    return x
# def top_k(x, k):
#     NINF = tf.cast(-100000000, x.dtype)
#     results = []
#     v = tf.reduce_max(x, axis=-1, keepdims=True)
#     results.append(v)
#     for i in range(k - 1):
#         x = tf.where(x == v, NINF, x)
#         v = tf.reduce_max(x, axis=-1, keepdims=True)
#         results.append(v)
#     x = tf.concat(results, axis=-1)
#     return x


def top_k(x, k):
    """Equivalent to tf.math.top_k(x, k) but more efficient on tpu."""
    last_dim_size = x.shape[-1]
    min_value = tf.math.reduce_min(x) - 1.0

    out_values = []

    for unused_i in range(k):
        index = tf.math.argmax(x, axis=-1, output_type=tf.int32)
        mask = tf.one_hot(index, last_dim_size, dtype=x.dtype)
        # TODO(yonghui): Would tf.gather be more efficient and numerically stable here?
        value = tf.reduce_sum(mask * x, -1, keepdims=True)
        x = (1.0 - mask) * x + mask * min_value
        out_values.append(value)

    return tf.concat(out_values, -1)


def cpu_variable(value, dtype):
    with tf.device("/device:CPU:0"):
        v = tf.Variable(value, trainable=False, dtype=dtype)
    return v