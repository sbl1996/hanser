import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf
from hanser.models.layers import Pool2d


def test_impl(size, kernel_size, stride):
    h = w = size
    x = tf.random.normal([2, h, w, 3])

    m = Pool2d(kernel_size, stride, padding='same', type='max')
    # m = tf.keras.layers.MaxPool2D(kernel_size, stride, padding='same')
    y = m(x)

    padding = (kernel_size - 1) // 2
    xt = torch.from_numpy(np.transpose(x, [0, 3, 1, 2]))
    mt = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=False)
    yt = mt(xt).detach().permute(0, 2, 3, 1)

    np.testing.assert_allclose(yt.numpy(), y.numpy(), atol=1e-6)


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
