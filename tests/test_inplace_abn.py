import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

from hanser.models.inplace_abn import InplaceABN
from hanser.train.optimizers import SGD

def copy_bn(l1, l2):
    l2.gamma.assign(l1.gamma)
    l2.beta.assign(l1.beta)
    l2.moving_mean.assign(l1.moving_mean)
    l2.moving_variance.assign(l1.moving_variance)


def train_batch(x, model, optimizer):
    with tf.GradientTape() as tape:
        y = model(x, training=True)
        y = tf.reduce_mean(y, axis=(1, 2))
        loss = tf.reduce_sum(y)
    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return y, grads


def test_inplace_abn():
    input_shape = (None, 8, 8, 4)

    m1 = Sequential([
        Conv2D(6, kernel_size=1, strides=1, padding='VALID', use_bias=False),
        BatchNormalization(momentum=0.9, epsilon=1e-3, fused=True),
        LeakyReLU(alpha=0.1),
        Conv2D(6, kernel_size=1, strides=1, padding='VALID', use_bias=False),
        BatchNormalization(momentum=0.9, epsilon=1e-3, fused=True),
        LeakyReLU(alpha=0.1),
    ])

    m2 = Sequential([
        Conv2D(6, kernel_size=1, strides=1, padding='VALID', use_bias=False),
        InplaceABN(momentum=0.9, epsilon=1e-3, alpha=0.1),
        Conv2D(6, kernel_size=1, strides=1, padding='VALID', use_bias=False),
        InplaceABN(momentum=0.9, epsilon=1e-3, alpha=0.1),
    ])

    optimizer1 = SGD(0.01, momentum=0.9, weight_decay=1e-4)
    optimizer2 = SGD(0.01, momentum=0.9, weight_decay=1e-4)

    m1.build(input_shape)
    m2.build(input_shape)

    m2.layers[0].kernel.assign(m1.layers[0].kernel)
    copy_bn(m2.layers[1], m1.layers[1])
    m2.layers[2].kernel.assign(m1.layers[3].kernel)
    copy_bn(m2.layers[3], m1.layers[4])

    for i in range(100):
        x = tf.random.stateless_normal((16, *input_shape[1:]), seed=(i, i), dtype=tf.float32)

        y1, grads1 = train_batch(x, m1, optimizer1)
        y2, grads2 = train_batch(x, m2, optimizer2)

    np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-6, atol=1e-6)

    for g1, g2 in zip(grads1, grads2):
        np.testing.assert_allclose(g1.numpy(), g2.numpy(), rtol=1e-6, atol=1e-6)