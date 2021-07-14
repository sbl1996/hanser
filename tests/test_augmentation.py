import math

from PIL import Image

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from hanser.transform import random_erasing, IMAGENET_MEAN, IMAGENET_STD, cutout, cutout2, cutout3
from hanser.transform.autoaugment.cifar import autoaugment

def grid_mask_batch(images, d_min=24, d_max=32, rotate=0, ratio=0.4, p=0.8):

    n, h, w = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2]
    hh = tf.cast(tf.math.sqrt(
        tf.cast(h * h + w * w, tf.float32)), tf.int32)  # to ensure coverage after rotation

    d = tf.cast(tf.random.uniform((n, 1), d_min, d_max + 1, dtype=tf.int32), tf.float32)  # the length of a unit's edge
    d_x = tf.cast(tf.math.ceil(tf.random.uniform((n, 1), tf.zeros_like(d), d)), dtype=tf.int32)  # bias
    d_y = tf.cast(tf.math.ceil(tf.random.uniform((n, 1), tf.zeros_like(d), d)), dtype=tf.int32)
    l = tf.cast(d * ratio, tf.int32)
    d = tf.cast(d, tf.int32)

    # generate masks
    idx = tf.repeat(tf.expand_dims(tf.range(hh), 0), repeats=n, axis=0)
    masks_x = tf.where(tf.math.logical_and(0 <= (idx - d_x) % d, (idx - d_x) % d < l),
                       tf.ones_like(idx),
                       tf.zeros_like(idx))
    masks_y = tf.where(tf.math.logical_and(0 <= (idx - d_y) % d, (idx - d_y) % d < l),
                       tf.ones_like(idx),
                       tf.zeros_like(idx))
    masks = tf.matmul(tf.expand_dims(masks_y, -1), tf.expand_dims(masks_x, 1))
    masks = tf.where(masks == 0, tf.ones_like(masks), tf.zeros_like(masks))

    masks = tf.repeat(tf.expand_dims(masks, -1), repeats=3, axis=-1)

    # rotate
    angles = tf.random.uniform((n,), 0, rotate, dtype=tf.float32)
    masks = tfa.image.rotate(masks, angles)

    # get the center part
    masks = masks[:, (hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w, :]

    # probability
    masks = tf.where(tf.random.uniform((n, 1, 1, 1)) < p, masks, tf.ones_like(masks))

    images *= tf.cast(masks, images.dtype)

    return images


im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))
# xc = x[:, 66:234:]
x = tf.cast(x, dtype=tf.float32)
x = (x - IMAGENET_MEAN) / IMAGENET_STD
# x2 = cutout2(x, 112, 'uniform')
xs = tf.stack([x, x, x], axis=0)
xs2 = grid_mask_batch(xs, 96, 160)
x2 = xs2[0]
# x2 = shear_x(xt, 1, 0)
x2 = x2 * IMAGENET_STD + IMAGENET_MEAN
x2 = x2.numpy()
x2 = x2.astype(np.uint8)
Image.fromarray(x2).show()