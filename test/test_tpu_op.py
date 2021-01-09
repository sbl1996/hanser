import numpy as np

import torch
import torch.nn as nn
import tensorflow as tf
from hanser.models.layers import Conv2d
from hanser.tpu import get_colab_tpu
from horch.nn import Swish

strategy = get_colab_tpu()

def t2t(x):
    if tf.is_tensor(x):
        return torch.from_numpy(x.numpy())
    elif torch.is_tensor(x):
        return tf.convert_to_tensor(x.numpy())
    else:
        raise ValueError("Not supported")


def t2np(x):
    if isinstance(x, (tuple, list)):
        return x.__class__(t2np(t) for t in x)
    elif torch.is_tensor(x):
        return x.detach().numpy()
    elif tf.is_tensor(x):
        return x.numpy()
    else:
        raise ValueError("Not supported")



def run_tf(fn, *args):
    if strategy is None:
        return fn(*args)
    else:
        strategy.run(fn, args)

def run_torch(fn, *args):
    return fn(*args)

l_tf = tf.keras.Sequential([
    Conv2d(2, 3, kernel_size=3, padding='same'),
    tf.keras.layers.Activation('swish'),
])
l_tf.build((None, 4, 4, 2))

l_torch = nn.Sequential(
    nn.Conv2d(2, 3, kernel_size=3, padding=1),
    Swish(),
)
with torch.no_grad():
    w = t2t(l_tf.layers[0].layers[1].kernel)
    b = t2t(l_tf.layers[0].layers[1].bias)
    l_torch[0].weight.copy_(w.permute(3, 2, 0, 1))
    l_torch[0].bias.copy_(b)


@tf.function
def test_train_tf(x):
    with tf.GradientTape() as tape:
        y = l_tf(x, training=True)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, l_tf.trainable_variables)
    return y, loss, grads


def test_train_torch(x):
    l_torch.zero_grad(True)
    l_torch.train()
    y = l_torch(x)
    loss = y.mean()
    loss.backward()
    grads = [p.grad for p in l_torch.parameters()]
    return y, loss, grads

x_tf = tf.random.normal((2, 4, 4, 2))
x_torch = t2t(x_tf).permute(0, 3, 1, 2)
y_tf = run_tf(test_train_tf, x_tf)
y_torch = run_torch(test_train_torch, x_torch)

y1, loss1, grads1 = t2np(y_tf)
y2, loss2, grads2 = t2np(y_torch)
np.testing.assert_allclose(y1, np.transpose(y2, [0, 2, 3, 1]), rtol=1e-5)

np.testing.assert_allclose(grads1[0], np.transpose(grads2[0], [2, 3, 1, 0]), rtol=1e-5)