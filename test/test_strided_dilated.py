import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

from hanser.models.layers import Conv2d

# Conclusion: BUG
# https://github.com/tensorflow/tensorflow/issues/31514#event-2551373576

w = np.full([3, 3, 2, 1], 1.0, dtype=np.float32)

m1 = Conv2d(2, 2, 3, stride=2, padding='valid', groups=2, dilation=2, bias=False)
m2 = Conv2d(2, 2, 3, stride=1, padding='valid', groups=2, dilation=2, bias=False)
m1.build((None, 4, 4, 2))
m2.build((None, 4, 4, 2))
m1.depthwise_kernel.assign(tf.convert_to_tensor(w))
m2.depthwise_kernel.assign(tf.convert_to_tensor(w))

x = tf.reshape(tf.range(64, dtype=tf.float32), [2, 4, 4, 2])
y1 = m1(tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]]))
y2 = m2(tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]]))


mt1 = nn.Conv2d(2, 2, 3, stride=2, padding=2, groups=2, dilation=2, bias=False)
mt2 = nn.Conv2d(2, 2, 3, stride=2, padding=2, groups=2, bias=False)
wt = torch.from_numpy(np.transpose(w, [2, 3, 0, 1]))
with torch.no_grad():
    mt1.weight.copy_(wt)
    mt2.weight.copy_(wt)

xt = torch.from_numpy(np.transpose(x.numpy(), [0, 3, 1, 2]))
yt1 = mt1(xt)
yt2 = mt2(xt)

yt1.detach().permute(0, 2, 3, 1).numpy()