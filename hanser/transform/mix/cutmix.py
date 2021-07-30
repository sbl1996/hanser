from toolz import curry

import tensorflow as tf
from hanser.transform.common import image_dimensions, _is_batch, _wrap_batch, _unwrap_batch
from hanser.transform.mix.common import _get_lam


def _rand_bbox(h, w, lam):
    r"""
    Note: lam may be (1,) or (n,)
    """
    cut_rat = tf.sqrt(1. - lam)
    cut_w = tf.cast(tf.cast(w, lam.dtype) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(h, lam.dtype) * cut_rat, tf.int32)

    cx = tf.random.uniform(tf.shape(lam), 0, w, dtype=tf.int32)
    cy = tf.random.uniform(tf.shape(lam), 0, h, dtype=tf.int32)

    l = tf.clip_by_value(cx - cut_w // 2, 0, w)
    t = tf.clip_by_value(cy - cut_h // 2, 0, h)
    r = tf.clip_by_value(cx + cut_w // 2, 0, w)
    b = tf.clip_by_value(cy + cut_h // 2, 0, h)

    return l, t, r, b


def _rand_mask(image, lam):
    n, h, w = image_dimensions(image, 4)[:3]
    l, t, r, b = _rand_bbox(h, w, lam)
    hi = tf.range(h)[None, :, None, None]
    mh = (hi >= t[:, None, None, None]) & (hi < b[:, None, None, None])
    wi = tf.range(w)[None, None, :, None]
    mw = (wi >= l[:, None, None, None]) & (wi < r[:, None, None, None])
    masks = tf.cast(tf.logical_not(mh & mw), image.dtype)
    lam = 1 - (b - t) * (r - l) / (h * w)
    return masks, lam


@curry
def cutmix_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = image_dimensions(image, 4)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    masks, lam = _rand_mask(image, lam)

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    image = image * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label.dtype)[:, None]
    label = label * lam + label2 * (1. - lam)
    return image, label


@curry
def cutmix_in_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0] // 2
    lam_shape = (n,) if hard else ()
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    masks, lam = _rand_mask(image1, lam)

    image = image1 * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label.dtype)[:, None]
    label = label1 * lam + label2 * (1. - lam)
    return image, label


@curry
def cutmix(data1, data2, alpha, hard=False, **gen_lam_kwargs):
    image1, label1 = data1
    image2, label2 = data2

    is_batch = _is_batch(image1)
    image1, label1, image2, label2 = _wrap_batch([
        image1, label1, image2, label2
    ], is_batch)

    n = image_dimensions(image1, 4)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    masks, lam = _rand_mask(image1, lam)

    image = image1 * masks + image2 * (1. - masks)

    lam = tf.cast(lam, label1.dtype)[:, None]
    label = label1 * lam + label2 * (1. - lam)

    image, label = _unwrap_batch([image, label], is_batch)
    return image, label