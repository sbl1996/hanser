import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

from hanser.models.inplace_abn import InplaceABN
from hanser.train.optimizers import SGD

x = tf.random.normal((16, 8, 8, 4), dtype=tf.float32)

input_shape = (None, 8, 8, 4)

m1 = Sequential([
    Conv2D(6, kernel_size=1, strides=1, padding='VALID', bias_initializer='normal'),
    BatchNormalization(momentum=0.9, epsilon=1e-3, fused=True),
    LeakyReLU(alpha=0.01),
])

m2 = Sequential([
    Conv2D(6, kernel_size=1, strides=1, padding='VALID', bias_initializer='normal'),
    InplaceABN(momentum=0.9, epsilon=1e-3, alpha=0.01),
])

optimizer1 = SGD(0.01, momentum=0.9, weight_decay=1e-4)
optimizer2 = SGD(0.01, momentum=0.9, weight_decay=1e-4)


m1.build(input_shape)
m2.build(input_shape)

m2.layers[0].kernel.assign(m1.layers[0].kernel)
m2.layers[0].bias.assign(m1.layers[0].bias)
m2.layers[1].gamma.assign(m1.layers[1].gamma)
m2.layers[1].beta.assign(m1.layers[1].beta)
m2.layers[1].moving_mean.assign(m1.layers[1].moving_mean)
m2.layers[1].moving_variance.assign(m1.layers[1].moving_variance)

for _i in range(100):
    with tf.GradientTape() as tape:
        y1 = m1(x, training=True)
        loss1 = tf.reduce_sum(y1)
    vars1 = m1.trainable_variables
    grads1 = tape.gradient(loss1, vars1)
    optimizer1.apply_gradients(zip(grads1, vars1))

for _i in range(100):
    with tf.GradientTape() as tape:
        y2 = m2(x, training=True)
        loss2 = tf.reduce_sum(y2)
    vars2 = m2.trainable_variables
    grads2 = tape.gradient(loss2, vars2)
    optimizer2.apply_gradients(zip(grads2, vars2))