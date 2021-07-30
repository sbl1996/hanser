from toolz import curry

import tensorflow as tf
from hanser.transform.common import image_dimensions, _is_batch, _wrap_batch, _unwrap_batch
from hanser.transform.mix.common import _get_lam


def _mixup(image1, label1, image2, label2, lam):

    lam_image = tf.cast(lam, image1.dtype)[:, None, None, None]
    image = lam_image * image1 + (1 - lam_image) * image2

    lam_label = tf.cast(lam, label1.dtype)[:, None]
    label = lam_label * label1 + (1 - lam_label) * label2
    return image, label


@curry
def mixup_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    indices = tf.random.shuffle(tf.range(n))
    image2 = tf.gather(image, indices)
    label2 = tf.gather(label, indices)

    lam_image = tf.cast(lam, image.dtype)[:, None, None, None]
    image = lam_image * image + (1 - lam_image) * image2

    lam_label = tf.cast(lam, label.dtype)[:, None]
    label = lam_label * label + (1 - lam_label) * label2
    return image, label


@curry
def mixup_in_batch(image, label, alpha, hard=False, **gen_lam_kwargs):
    n = tf.shape(image)[0] // 2
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)
    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    return _mixup(image1, label1, image2, label2, lam)


@curry
def mixup(data1, data2, alpha, hard=False, **gen_lam_kwargs):
    image1, label1 = data1
    image2, label2 = data2

    is_batch = _is_batch(image1)
    image1, label1, image2, label2 = _wrap_batch([
        image1, label1, image2, label2
    ], is_batch)

    n = image_dimensions(image1, 4)[0]
    lam_shape = (n,) if hard else (1,)
    lam = _get_lam(lam_shape, alpha, **gen_lam_kwargs)

    image, label = _mixup(image1, label1, image2, label2, lam)

    image, label = _unwrap_batch([image, label], is_batch)
    return image, label
