import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense

h = 3
w = 3
c_in = 3
c_out = 4
l1 = Conv2D(c_out, kernel_size=1, strides=1, use_bias=False)
l2 = Dense(c_out, use_bias=False)

l1.build((None, h, w, c_in))
l2.build((None, h, w, c_in))

l2.kernel.assign(l1.kernel.numpy()[0, 0])

x = tf.random.normal((2, h, w, c_in))

y1 = l1(x)
y2 = l2(x)

np.testing.assert_allclose(y1.numpy(), y2.numpy())