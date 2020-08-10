import numpy as np

import torch
import torch.nn as nn

import tensorflow as tf

x = tf.random.normal([2, 8, 8, 2])

m = tf.keras.layers.MaxPool2D(3, 2, padding='same')
y = m(x)

xt = torch.from_numpy(np.transpose(x, [0, 3, 1, 2]))
mt = nn.MaxPool2d(3, 2, padding=1, ceil_mode=False)
yt = mt(xt)
yt = yt.detach().permute(0, 2, 3, 1)




m = tf.keras.layers.AvgPool2D(3, 2, padding='same')
y = m(x)

xt = torch.from_numpy(np.transpose(x, [0, 3, 1, 2]))
mt = nn.AvgPool2d(3, 2, padding=1, ceil_mode=False, count_include_pad=False)
yt = mt(xt)
yt = yt.detach().permute(0, 2, 3, 1)