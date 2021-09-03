import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

from hanser.models.inplace_abn import InplaceABN

x = tf.random.normal((16, 1, 1, 4), dtype=tf.float32)

input_shape = (None, 1, 1, 4)

m1 = Sequential([
    Conv2D(6, kernel_size=1, strides=1, padding='VALID', bias_initializer='normal'),
    BatchNormalization(momentum=0.9, epsilon=1e-5, fused=True),
    LeakyReLU(alpha=0.01),
])

m2 = Sequential([
    Conv2D(6, kernel_size=1, strides=1, padding='VALID', bias_initializer='normal'),
    InplaceABN(momentum=0.9, epsilon=1e-5, alpha=0.01),
])

m1.build(input_shape)
m2.build(input_shape)

m2.layers[0].kernel.assign(m1.layers[0].kernel)
m2.layers[0].bias.assign(m1.layers[0].bias)
m2.layers[1].gamma.assign(m1.layers[1].gamma)
m2.layers[1].beta.assign(m1.layers[1].beta)
m2.layers[1].moving_mean.assign(m1.layers[1].moving_mean)
m2.layers[1].moving_variance.assign(m1.layers[1].moving_variance)


with tf.GradientTape() as tape:
    y1 = m1(x, training=True)
    loss1 = tf.reduce_sum(y1)
gs1 = tape.gradient(loss1, m1.trainable_variables)


with tf.GradientTape() as tape:
    y2 = m2(x, training=True)
    loss2 = tf.reduce_sum(y2)
gs2 = tape.gradient(loss2, m2.trainable_variables)

d = [g1 - g2 for g1, g2 in zip(gs1, gs2)]
print(d)