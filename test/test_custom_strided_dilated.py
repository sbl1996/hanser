import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d


def test_impl(kernel_size=3, size=4, dilation=2, channels=2):
    h = w = size
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

    x1 = tf.random.normal([2, h, w, channels])

    m = Conv2d(channels, channels, kernel_size, stride=2, groups=channels, dilation=2, bias=False)
    m.build((None, h, w, channels))

    y1 = m(x1)

    mt = nn.Conv2d(channels, channels, kernel_size, stride=2,
                   padding=padding, groups=channels, dilation=dilation, bias=False)
    with torch.no_grad():
        mt.weight.copy_(torch.from_numpy(np.transpose(m.depthwise_kernel.numpy(), [2, 3, 0, 1])))

    xt1 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))
    yt1 = mt(xt1).detach().permute(0, 2, 3, 1)

    np.testing.assert_allclose(yt1.numpy(), y1.numpy(), atol=1e-7)


kernel_sizes = [3, 5]
dilations = [2, 3, 4]
sizes = [4, 5, 8]

for kernel_size in kernel_sizes:
    for size in sizes:
        for dilation in dilations:
            test_impl(kernel_size, size, 2, 32)
