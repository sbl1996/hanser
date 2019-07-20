import tensorflow as tf


def random_crop(x, size, padding, fill=128):
    height, width = size
    ph, pw = padding
    x = tf.pad(x, [(ph, ph), (pw, pw), (0, 0)], constant_values=fill)
    x = tf.image.random_crop(x, [height, width, x.shape[-1]])
    return x


def cutout(image, length):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    cy = tf.random_uniform((), 0, h, dtype=tf.int32)
    cx = tf.random_uniform((), 0, w, dtype=tf.int32)

    t = tf.maximum(0, cy - length // 2)
    b = tf.minimum(h, cy + length // 2)
    l = tf.maximum(0, cx - length // 2)
    r = tf.minimum(w, cx + length // 2)
    shape = [b - t, r - l]
    padding = [(t, h - b), (l, w - r)]

    mask = tf.pad(tf.zeros(shape, dtype=image.dtype), padding, constant_values=1)

    mask = tf.expand_dims(mask, -1)
    # mask = tf.tile(mask, [1, 1, 3])
    # image = tf.where(
    #     tf.equal(mask, 0),
    #     tf.ones_like(image, dtype=image.dtype)
    # )
    image = image * mask
    return image