from toolz import curry
import tensorflow as tf

from hanser.transform.mix.mixup import mixup, mixup_batch, mixup_in_batch
from hanser.transform.mix.cutmix import cutmix, cutmix_batch, cutmix_in_batch
from hanser.transform.mix.resizemix import resizemix_batch
from hanser.transform.mix.fmix import fmix
from hanser.transform.common import image_dimensions


@curry
def mixup_or_cutmix_batch(
    image, label, mixup_alpha=0.2, cutmix_alpha=1.0, switch_prob=0.5, hard=False, **gen_lam_kwargs):
    if tf.random.uniform(()) < switch_prob:
        return mixup_batch(image, label, mixup_alpha, hard=hard, **gen_lam_kwargs)
    else:
        return cutmix_batch(image, label, cutmix_alpha, hard=hard, **gen_lam_kwargs)


@curry
def mixup_cutmix_batch(
    image, label, mixup_alpha=0.2, cutmix_alpha=1.0, hard=False, **gen_lam_kwargs):
    n = image_dimensions(image, 4)[0] // 2
    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    image1, label1 = mixup_batch(image1, label1, mixup_alpha, hard=hard, **gen_lam_kwargs)
    image2, label2 = cutmix_batch(image2, label2, cutmix_alpha, hard=hard, **gen_lam_kwargs)
    image = tf.concat((image1, image2), axis=0)
    label = tf.concat((label1, label2), axis=0)
    return image, label


@curry
def mixup_cutmix_batch2(
    image, label, mixup_alpha=0.2, cutmix_alpha=1.0, hard=False, **gen_lam_kwargs):
    n = image_dimensions(image, 4)[0] // 2
    image1, image2 = image[:n], image[n:]
    label1, label2 = label[:n], label[n:]
    data1, data2 = (image1, label1), (image2, label2)
    image1, label1 = mixup(data1, data2, mixup_alpha, hard=hard, **gen_lam_kwargs)
    image2, label2 = cutmix(data1, data2, cutmix_alpha, hard=hard, **gen_lam_kwargs)
    image = tf.concat((image1, image2), axis=0)
    label = tf.concat((label1, label2), axis=0)
    return image, label