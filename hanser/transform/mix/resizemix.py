import tensorflow as tf
from hanser.transform.common import image_dimensions


def _random_int(shape, minval, maxval):
    minval = tf.cast(minval, tf.float32)
    maxval = tf.cast(maxval, tf.float32)
    xs = tf.random.uniform(shape, minval, maxval, dtype=tf.float32)
    xs = tf.cast(xs, tf.int32)
    return xs


def _sample(shape, scale, sample_area):
    if sample_area:
        area = tf.random.uniform(shape, scale[0], scale[1], dtype=tf.float32)
        tau = tf.sqrt(area)
    else:
        tau = tf.random.uniform(shape, scale[0], scale[1], dtype=tf.float32)
        area = tau ** 2
    return tau, area


def _resize_and_mix(args):
    image, image2, h_t, w_t, l, t, r, b = args
    is_batch = len(image.shape) == 4
    dtype = image.dtype

    image1 = tf.image.resize(image, (h_t, w_t), method='bilinear')
    if dtype == tf.uint8:
        image1 = tf.cast(image1, dtype)

    top = image2[..., :t, :, :]
    mid_left = image2[..., t:b, :l, :]
    mid_right = image2[..., t:b, r:, :]
    bottom = image2[..., b:, :, :]

    mid = tf.concat([mid_left, image1, mid_right], 2 if is_batch else 1)
    image = tf.concat([top, mid, bottom], 1 if is_batch else 0)
    return image


def resizemix_batch(image, label, scale=(0.1, 0.8), hard=False, sample_area=False):

    n, h, w, c = image_dimensions(image, 4)
    shape = (n,) if hard else ()
    tau, area = _sample(shape, scale, sample_area)
    h_t = tf.cast(tf.cast(h, tf.float32) * tau, tf.int32)
    w_t = tf.cast(tf.cast(w, tf.float32) * tau, tf.int32)

    h_t_1, h_t_2 = h_t // 2, h_t - h_t // 2
    w_t_1, w_t_2 = w_t // 2, w_t - w_t // 2
    cx = _random_int(shape, w_t_1, w - w_t_2)
    cy = _random_int(shape, h_t_1, h - h_t_2)
    l, t, r, b = cx - w_t_1, cy - h_t_1, cx + w_t_2, cy + h_t_2

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    if hard:
        image = tf.map_fn(
            _resize_and_mix, (image, image2, h_t, w_t, l, t, r, b), fn_output_signature=image.dtype)
    else:
        image = _resize_and_mix((image, image2, h_t, w_t, l, t, r, b))

    image.set_shape((n, h, w, c))

    lam = area[:, None] if hard else area
    label = label * lam + label2 * (1. - lam)
    return image, label