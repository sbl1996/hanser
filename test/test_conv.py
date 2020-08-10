import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d

kernel_size = 3
size = 4
channels = 2

h = w = size

x1 = tf.random.normal([1, h, w, channels])


m = Conv2d(channels, channels, kernel_size, stride=2,
           padding='VALID', groups=channels, bias=False)
m.build((None, h, w, channels))

y1 = m(tf.pad(x1, [(0, 0), (1, 0), (1, 0), (0, 0)]))

padding = (kernel_size - 1) // 2
mt = nn.Conv2d(channels, channels, kernel_size, stride=2,
               padding=padding, groups=channels, bias=False)
with torch.no_grad():
    mt.weight.copy_(torch.from_numpy(np.transpose(m.depthwise_kernel.numpy(), [2, 3, 0, 1])))

xt1 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))
yt1 = mt(xt1).detach().permute(0, 2, 3, 1)

np.testing.assert_allclose(yt1.numpy(), y1.numpy(), atol=1e-6)
