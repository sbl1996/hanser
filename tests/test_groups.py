import numpy as np
import tensorflow as tf

from hanser.models.layers import Conv2d

l = Conv2d(3, 6, 1, groups=3, bias=False)
x = tf.random.normal([1, 2, 2, 3])
l(x)

w = l.weights[0]
wa = w.numpy()
wa[:, :, 0] *= 100 * np.abs(wa[:, :, 0])
wa[:, :, 1] *= 0
wa[:, :, 2] = 100000 * np.abs(wa[:, :, 2])
w.assign(wa)

y = l(x).numpy()
y[:, :, :, 0:2]
y[:, :, :, 2:4]
y[:, :, :, 4:6]
