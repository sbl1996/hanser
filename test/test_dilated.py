import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d

channels = 32
kernel_size = 5
dilation = 2
h = w = 4
padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

m = Conv2d(channels, channels, kernel_size, stride=1,
           padding='same', groups=channels, dilation=dilation, bias=False)
m.build((None, h, w, channels))

x1 = tf.random.normal([2, h, w, channels])
y1 = m(x1)


mt = nn.Conv2d(channels, channels, kernel_size, stride=1,
               padding=padding, groups=channels, dilation=dilation, bias=False)
with torch.no_grad():
    mt.weight.copy_(torch.from_numpy(np.transpose(m.depthwise_kernel.numpy(), [2, 3, 0, 1])))

xt1 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))
yt1 = mt(xt1)

np.testing.assert_allclose(yt1.detach().permute(0, 2, 3, 1).numpy(), y1.numpy(), atol=1e-7)