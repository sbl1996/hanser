import math

import numpy as np
from toolz import curry

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics as M

from hanser.losses import CrossEntropy
from hanser.models.layers import DEFAULTS
from hanser.models.darts.model_search import Network
from hanser.models.darts.genotypes import set_primitives
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10_tfds
from hanser.train.nas.trainer import Trainer
from hanser.transform import random_crop, cutout, normalize, to_tensor
from hanser.train.lr_schedule import CosineLR
from hanser.transform.autoaugment import autoaugment
from hanser.io import time_now

@curry
def preprocess(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    label = tf.one_hot(label, 10)
    # image = tf.cast(image, tf.bfloat16)

    if training:
        image = cutout(image, 16)

    return image, label


(x_train, y_train), (x_test, y_test) = load_cifar10_tfds()

train_portion = 0.5
n_train = 50000
n_search = int(n_train * (1 - train_portion))
n_train = n_train - n_search
n_test = 10000

x_search, y_search = x_train[n_search:], y_train[n_search:]
x_train, y_train = x_train[:n_train], y_train[:n_train]

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_search = tf.data.Dataset.from_tensor_slices((x_search, y_search))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

mul = 8
num_train_examples = 250
num_search_examples = 250
num_test_examples = 100
ds_train = ds_train.take(num_train_examples)
ds_search = ds_search.take(num_search_examples)
ds_test = ds_test.take(num_test_examples)

batch_size = 2 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds_train = prepare(ds_train, preprocess(training=True), batch_size, training=True, buffer_size=10000)
ds_search = prepare(ds_search, preprocess(training=True), batch_size, training=True, buffer_size=10000)
ds_test = prepare(ds_test, preprocess(training=False), eval_batch_size, training=False)

DEFAULTS['affine'] = False
set_primitives('tiny')
model = Network(4, 5, 4, 4, 3, 10)
input_shape = (32, 32, 3)
model.build((None, *input_shape))
# model.call(Input(input_shape))

criterion = CrossEntropy(reduction='none')

base_lr = 0.001
epochs = 20
optimizer_arch = Adam(3e-4, beta_1=0.5)
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=1e-5, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer_model = SGD(lr_shcedule, momentum=0.9, nesterov=True)

metrics = [
    M.Mean(name='loss'), M.CategoricalAccuracy(name='acc')]
test_metrics = [
    M.CategoricalCrossentropy(name='loss', from_logits=True), M.CategoricalAccuracy(name='acc')]
# metric_transform = lambda x: x[0]

trainer = Trainer(model, criterion, optimizer_arch, optimizer_model,
                  metrics, test_metrics, 5.0, 1e-3, 3e-4)

print(time_now())
trainer.fit(epochs, ds_train, ds_search, steps_per_epoch, ds_test, test_steps)
