import numpy as np

import torch

import tensorflow as tf

grad_clip_norm = 5.0

xs = [tf.random.normal([2, 3, 4]) * 5 for _ in range(10)]

xts = []
for x in xs:
    xt = torch.from_numpy(x.numpy())
    xt.grad = xt.clone()
    xts.append(xt)

xs_clip1 = [tf.clip_by_norm(x, grad_clip_norm) for x in xs]
xs_clip2, global_norm = tf.clip_by_global_norm(xs, grad_clip_norm)

global_norm_t = torch.nn.utils.clip_grad_norm_(xts, grad_clip_norm)
xts_clip = [x.grad for x in xts]

for x1, x2 in zip(xts_clip, xs_clip2):
    x1 = x1.numpy()
    x2 = x2.numpy()
    np.testing.assert_array_almost_equal(x1, x2)