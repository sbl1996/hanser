import os

import math
from tqdm import tqdm
from toolz import curry
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow_addons.optimizers import AdamW
from hanser.datasets import prepare
from hanser.transform import to_tensor, normalize
from hanser.distribute import setup_runtime, distribute_datasets

from hanser.models.layers import Conv2d, Linear
from hanser.ops import gumbel_softmax

@curry
def transform(x, y, training):
    x, y = to_tensor(x, y, vmax=16)
    x = normalize(x, [4.8842], [6.0168])
    return x, y

X, y = load_digits(n_class=10, return_X_y=True)
X = X.reshape(-1, 8, 8, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

batch_size = 32
eval_batch_size = batch_size * 2
n_train, n_test = len(X_train), len(X_test)
steps_per_epoch, test_steps = n_train // batch_size, math.ceil(n_test / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

ds_train = prepare(ds, batch_size, transform=transform(training=True), training=True, buffer_size=n_train)
ds_test = prepare(ds_test, eval_batch_size, transform=transform(training=False), training=False)

setup_runtime()
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

class MixOp(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.ops = [
            Conv2d(4, 4, kernel_size=1, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=3, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=3, groups=4, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=5, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=5, groups=4, norm='def', act='def'),
        ]

        self.alpha = self.add_weight(
            name='alpha', shape=(len(self.ops),), dtype=self.dtype,
            initializer=tf.keras.initializers.RandomNormal(stddev=1e-1), trainable=True,
        )

    def _create_branch_fn(self, x, i, hardwts):
        return lambda: self.ops[i](x) * hardwts[i]

    def build(self, input_shape):
        for op in self.ops:
            op.build(input_shape)

    def call(self, x):
        hardwts, index = gumbel_softmax(self.alpha, tau=1.0, hard=True, return_index=True)
        branch_fns = [
            self._create_branch_fn(x, i, hardwts)
            for i in range(3)
        ]
        return tf.switch_case(index, branch_fns)

class ConvNet(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stem = Conv2d(1, 4, kernel_size=1)
        self.op = MixOp()

        self.flatten = Flatten()
        self.fc = Linear(4, 10)
    
    def call(self, x):
        x = self.stem(x)
        x = self.op(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = ConvNet()
model.build((None, 8, 8, 1))
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

for i in range(50):
    it = iter(ds_train)
    for i in tqdm(range(steps_per_epoch)):
        x, y = next(it)

        with tf.GradientTape() as tape:
            p = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, p, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(tf.nn.softmax(model.op.alpha))

# weights = tf.Variable(tf.random.normal([3]), trainable=True)
# x = tf.random.normal([2, 8, 8, 4])
# op = MixOp()
# op.build((None, 8, 8, 4))

# with tf.GradientTape() as tape:
#     hardwts, index = gumbel_softmax(weights, tau=1.0, hard=True, return_index=True)
#     x = op(x, hardwts, index)
#     loss = tf.reduce_sum(x)

# grads = tape.gradient(loss, weights)
# grads

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