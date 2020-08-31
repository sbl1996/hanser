import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf
from hanser.models.layers import Pool2d

size = 4
kernel_size = 3
stride = 2
def test_impl(size, kernel_size, stride):
    h = w = size
    x = tf.random.normal([2, h, w, 3])

    m = Pool2d(kernel_size, stride, padding='same', type='max')
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = m(x)
        loss = tf.reduce_sum(y)
    g = tape.gradient(loss, [x])[0]

    padding = (kernel_size - 1) // 2
    xt = torch.from_numpy(np.transpose(x, [0, 3, 1, 2])).requires_grad_(True)
    mt = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=False)
    yt = mt(xt)
    loss_t = yt.sum()
    loss_t.backward()
    gt = xt.grad.permute(0, 2, 3, 1)

    d_f = yt.permute(0, 2, 3, 1).detach().numpy() - y.numpy()
    print(d_f.mean(), d_f.std())
    np.testing.assert_allclose(yt.permute(0, 2, 3, 1).detach().numpy(), y.numpy(), atol=1e-6)

    d_b = g.numpy() - gt.numpy()
    print(d_b.mean(), d_b.std())
    np.testing.assert_allclose(g.numpy(), gt.numpy(), atol=1e-6)



sizes = [4, 8, 16]
kernel_sizes = [2]
strides = [2]
# kernel_sizes = [3]
# strides = [1, 2]
for size in sizes:
    for k in kernel_sizes:
        for s in strides:
            try:
                test_impl(size, k, s)
            except AssertionError as e:
                print(size, k, s)
                raise e
