import tensorflow as tf

from hanser.models.layers import Conv2d

h = w = 3
c = 8
g = 4
c1 = c // g

l = Conv2d(c, c, 3, groups=g, bias=False)
l.build((None, h, w, c))
k = l.layers[1].kernel

x = tf.random.normal([1, h, w, c])
y = l(x)

l1 = Conv2d(c1, c1, 3, bias=False)
l1.build((None, h, w, c1))
k1 = l1.layers[1].kernel
k1.assign(k[:, :, :, :c1])
x1 = x[:, :, :, :c1]
y1 = l1(x1)