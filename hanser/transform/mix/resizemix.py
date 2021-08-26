from toolz import curry

import tensorflow as tf
from hanser.transform.common import image_dimensions

def _random_int(shape, minval, maxval):
    minval = tf.cast(minval, tf.float32)
    maxval = tf.cast(maxval, tf.float32)
    xs = tf.random.uniform(shape, minval, maxval, dtype=tf.float32)
    xs = tf.cast(xs, tf.int32)
    return xs

def _resizemix_batch_hard(image, label, scale=(0.1, 0.8)):

    n, h, w, c = image_dimensions(image, 4)
    tau = tf.random.uniform((n,), scale[0], scale[1], dtype=tf.float32)
    h_t = tf.cast(tf.cast(h, tf.float32) * tau, tf.int32)
    w_t = tf.cast(tf.cast(w, tf.float32) * tau, tf.int32)

    h_t_1, h_t_2 = h_t // 2, h_t - h_t // 2
    w_t_1, w_t_2 = w_t // 2, w_t - w_t // 2
    cx = _random_int((n,), w_t_1, w - w_t_2)
    cy = _random_int((n,), h_t_1, h - h_t_2)
    l, t, r, b = cx - w_t_1, cy - h_t_1, cx + w_t_2, cy + h_t_2

    def _resizemix(args):
        image, image2, h_t, w_t, l, t, r, b = args
        image1 = tf.image.resize(image, (h_t, w_t), method='bilinear')

        top = image2[:t, :, :]
        mid_left = image2[t:b, :l, :]
        mid_right = image2[t:b, r:, :]
        bottom = image2[b:, :, :]

        mid = tf.concat([mid_left, image1, mid_right], 1)
        image = tf.concat([top, mid, bottom], 0)
        image.set_shape((h, w, c))
        return image

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    image = tf.map_fn(
        _resizemix,(image, image2, h_t, w_t, l, t, r, b), fn_output_signature=image.dtype)
    lam = (tau ** 2)[:, None]
    label = label * lam + label2 * (1. - lam)
    return image, label


def resizemix_batch(image, label, scale=(0.1, 0.8), hard=False):
    if hard:
        return _resizemix_batch_hard(image, label, scale)

    n, h, w, c = image_dimensions(image, 4)

    tau = tf.random.uniform((), scale[0], scale[1], dtype=tf.float32)
    h_t = tf.cast(tf.cast(h, tf.float32) * tau, tf.int32)
    w_t = tf.cast(tf.cast(w, tf.float32) * tau, tf.int32)
    image1 = tf.image.resize(image, (h_t, w_t), method='bilinear')

    h_t_1, h_t_2 = h_t // 2, h_t - h_t // 2
    w_t_1, w_t_2 = w_t // 2, w_t - w_t // 2
    cx = tf.random.uniform((), w_t_1, w - w_t_2, dtype=tf.int32)
    cy = tf.random.uniform((), h_t_1, h - h_t_2, dtype=tf.int32)
    l, t, r, b = cx - w_t_1, cy - h_t_1, cx + w_t_2, cy + h_t_2

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    top = image2[:, :t, :, :]
    mid_left = image2[:, t:b, :l, :]
    mid_right = image2[:, t:b, r:, :]
    bottom = image2[:, b:, :, :]

    mid = tf.concat([mid_left, image1, mid_right], 2)
    image = tf.concat([top, mid, bottom], 1)
    image.set_shape((n, h, w, c))

    lam = tau ** 2
    label = label * lam + label2 * (1. - lam)
    return image, label
