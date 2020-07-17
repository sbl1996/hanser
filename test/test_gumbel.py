import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, Linear
import paddle.fluid.layers as L


def s(x):
    return x.numpy()


fluid.enable_dygraph()

def gumbel_sample(a, tau=1.0):
    o = -L.log(-L.log(L.uniform_random([n, *a.shape], min=0.0, max=1.0)))
    # l = to_variable(np.array([0.4, 0.4, 0.4]).astype('float32'))
    o = -L.log(-L.log(l))
    return L.softmax((L.log_softmax(a) + o) / tau)

l1 = Linear(2, 4)
l2 = Linear(2, 4)
l3 = Linear(2, 4)
a = L.create_parameter([3], 'float32')
a.set_value(np.array([1, 2, 3]).astype('float32'))
ops = [l1, l2, l3]

x = to_variable(np.array([
    [1, 2],
    [3, 4],
]).astype('float32'))
y = to_variable(np.array([
    [2, 2, 0, 0],
    [3, 1, 2, 3]
]).astype('float32'))

p = L.softmax(a)
ss = []
for i, op in enumerate(ops):
    ss.append(op(x) * p[i])
pred = sum(ss)
loss = L.mean(L.abs(pred - y))
loss.backward()
a.gradient()
# [ 0.04809267, -0.3332683 ,  0.28517565]


p = L.softmax(a)
index = 2
ss.append(ops[2] * p[2])
pred = sum(ss)
loss = L.mean(L.abs(pred - y))
loss.backward()
a.gradient()



p = L.softmax(a)
# index = np.random.choice(3, p=p.numpy())
index = 2
i = to_variable(np.array(index))
one_h = L.one_hot(L.unsqueeze(i, [0,1]), a.shape[0])[0]
hardwts = one_h - p.detach() + p
pred = sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(ops))
# loss = L.mean(L.abs(pred - y))
loss = L.mse_loss(pred, y)
loss.backward()
a.gradient()

p = gumbel_sample(a)
i = L.argmax(p)
index = int(i.numpy())
one_h = L.one_hot(L.unsqueeze(i, [0,1]), a.shape[0])[0]
hardwts = one_h - p.detach() + p
pred = sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(ops))
# loss = L.mean(L.abs(pred - y))
loss = L.mse_loss(pred, y)
loss.backward()
a.gradient()

l1.clear_gradients()
l2.clear_gradients()
l3.clear_gradients()
a.clear_gradient()