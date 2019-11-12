import tensorflow as tf


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