import tensorflow as tf


def ce(y, p):
    # y: (N, C)
    # p: (N, C)
    losses = -y * tf.math.log(p)
    return tf.reduce_mean(tf.reduce_sum(losses, axis=-1))


def bce(y, p):
    # y: (N, C)
    # p: (N, C)
    losses = -y * tf.math.log(p) - (1-y) * tf.math.log(1-p)
    return tf.reduce_mean(tf.reduce_mean(losses, axis=-1))


y = [
    [0.9, 0.05, 0.05],
]

p = [
    [0.6, 0.35, 0.05],
]

y = tf.convert_to_tensor(y)
p = tf.convert_to_tensor(p)
ce(y, p), bce(y, p), tf.keras.losses.categorical_crossentropy(y, p)

import torch
import torch.nn.functional as F

yt = torch.tensor(y.numpy())
pt = torch.tensor(p.numpy())
F.binary_cross_entropy(pt, yt, reduction='mean')