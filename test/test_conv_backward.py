import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d

# size = 9
# channels = 4
# kernel_size = 3
# stride = 1
# dilation = 2

def test_impl(size, channels, kernel_size, stride, dilation):
    h = w = size
    x1 = tf.random.normal([2, h, w, channels])

    m = Conv2d(channels, channels, kernel_size, stride=stride, padding='same',
               groups=channels, dilation=dilation, bias=False)
    m.build((None, h, w, channels))

    with tf.GradientTape() as tape:
        y1 = m(x1)
        loss = tf.reduce_sum(y1)
    g = tape.gradient(loss, m.trainable_variables)[0]


    if isinstance(m, tf.keras.Sequential):
        weight = m.layers[1].depthwise_kernel
    else:
        weight = m.depthwise_kernel
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    mt = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding,
                   groups=channels, dilation=dilation, bias=False)
    with torch.no_grad():
        mt.weight.copy_(torch.from_numpy(np.transpose(weight.numpy(), [2, 3, 0, 1])))

    xt1 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))

    yt1 = mt(xt1)
    loss_t = yt1.sum()
    loss_t.backward()
    gt = mt.weight.grad.permute(2, 3, 0, 1)

    d = g.numpy() - gt.numpy()
    print(d.mean(), d.std())
    np.testing.assert_allclose(g.numpy(), gt.numpy(), atol=1e-6)


sizes = [4, 7, 8, 16]
channels = 4
kernel_sizes = [1, 3, 5, 7]
strides = [1, 2]
dilations = [1, 2]
for size in sizes:
    for k in kernel_sizes:
        for s in strides:
            for d in dilations:
                try:
                    print(size, k, s, d)
                    test_impl(size, channels, k, s, d)
                except AssertionError as e:
                    print("Error: ", end="")
                    print(size, k, s, d)
                    raise e
