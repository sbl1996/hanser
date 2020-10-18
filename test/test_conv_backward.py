import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d

channels = 128
size = 8
kernel_size = 3
stride = 1
dilation = 1

def test_impl(size, channels, kernel_size, stride, dilation):
    h = w = size
    x1 = tf.random.normal([32, h, w, 2])

    m = Conv2d(2, channels, kernel_size, stride=stride, padding='same',
               groups=1, dilation=dilation, bias=False)
    m.build((None, h, w, 2))

    with tf.GradientTape() as tape:
        y1 = m(x1)
        loss = tf.reduce_sum(y1)
    g = tape.gradient(loss, m.trainable_variables)[0]

    if isinstance(m, tf.keras.Sequential):
        conv = m.layers[1]
    else:
        conv = m
    if "Depth" in str(type(conv)):
        weight = conv.depthwise_kernel
    else:
        weight = conv.kernel
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    mt = nn.Conv2d(2, channels, kernel_size, stride=stride, padding=padding,
                   groups=1, dilation=dilation, bias=False)
    with torch.no_grad():
        if "Depth" in str(type(conv)):
            T = [2, 3, 0, 1]
        else:
            T = [3, 2, 0, 1]
        mt.weight.copy_(torch.from_numpy(np.transpose(weight.numpy(), T)))

    xt1 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))

    yt1 = mt(xt1)
    loss_t = yt1.sum()
    loss_t.backward()
    gt = mt.weight.grad.numpy()
    g = np.transpose(g.numpy(), T)

    d = g - gt
    print(d.mean(), d.std())
    np.testing.assert_allclose(g.numpy(), gt.numpy(), atol=1e-6)


dilations = [1, 2]
sizes = [4, 7, 8, 13, 16]
channels = 4
kernel_sizes = [1, 3, 5, 7]
strides = [1, 2]
for d in dilations:
    for size in sizes:
        for k in kernel_sizes:
            for s in strides:
                try:
                    print(size, k, s, d)
                    test_impl(size, channels, k, s, d)
                    print()
                except AssertionError as e:
                    print("Error: ", end="")
                    print(size, k, s, d)
                    print()
