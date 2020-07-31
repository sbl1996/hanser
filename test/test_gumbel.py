import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from hanser.models.modules import Conv2d
from hanser.ops import gumbel_softmax

def s(x):
    return x.numpy()

class MixOp(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.ops = [
            Conv2d(4, 4, kernel_size=3, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=3, groups=4, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=5, groups=4, norm='def', act='def'),
        ]

    def _create_branch_fn(self, i, hardwts):
        return lambda: self.ops[i](x) * hardwts[i]

    def call(self, x, hardwts, index):
        branch_fns = [
            self._create_branch_fn(i, hardwts)
            for i in range(3)
        ]
        return tf.switch_case(index, branch_fns)

weights = tf.Variable(tf.random.normal([3]), trainable=True)
x = tf.random.normal([2, 8, 8, 4])
op = MixOp()

hardwts, index = gumbel_softmax(weights, tau=1.0, hard=True, return_index=True)
y = op(x, hardwts, index)

with tf.GradientTape() as tape:
    x = gumbel_softmax(logits, tau=1.0, hard=True)
    loss = tf.reduce_sum(x) * 1000

grads = tape.gradient(loss, logits)
grads

# l1 = Linear(2, 4)
# l2 = Linear(2, 4)
# l3 = Linear(2, 4)
# a = L.create_parameter([3], 'float32')
# a.set_value(np.array([1, 2, 3]).astype('float32'))
# ops = [l1, l2, l3]
#
# x = to_variable(np.array([
#     [1, 2],
#     [3, 4],
# ]).astype('float32'))
# y = to_variable(np.array([
#     [2, 2, 0, 0],
#     [3, 1, 2, 3]
# ]).astype('float32'))
#
# p = L.softmax(a)
# ss = []
# for i, op in enumerate(ops):
#     ss.append(op(x) * p[i])
# pred = sum(ss)
# loss = L.mean(L.abs(pred - y))
# loss.backward()
# a.gradient()
# # [ 0.04809267, -0.3332683 ,  0.28517565]
#
#
# p = L.softmax(a)
# index = 2
# ss.append(ops[2] * p[2])
# pred = sum(ss)
# loss = L.mean(L.abs(pred - y))
# loss.backward()
# a.gradient()
#
#
#
# p = L.softmax(a)
# # index = np.random.choice(3, p=p.numpy())
# index = 2
# i = to_variable(np.array(index))
# one_h = L.one_hot(L.unsqueeze(i, [0,1]), a.shape[0])[0]
# hardwts = one_h - p.detach() + p
# pred = sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(ops))
# # loss = L.mean(L.abs(pred - y))
# loss = L.mse_loss(pred, y)
# loss.backward()
# a.gradient()
#
# p = gumbel_sample(a)
# i = L.argmax(p)
# index = int(i.numpy())
# one_h = L.one_hot(L.unsqueeze(i, [0,1]), a.shape[0])[0]
# hardwts = one_h - p.detach() + p
# pred = sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(ops))
# # loss = L.mean(L.abs(pred - y))
# loss = L.mse_loss(pred, y)
# loss.backward()
# a.gradient()
#
# l1.clear_gradients()
# l2.clear_gradients()
# l3.clear_gradients()
# a.clear_gradient()