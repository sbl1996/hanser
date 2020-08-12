import math

import numpy as np
from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics as M
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10_tfds
from hanser.transform import random_crop, cutout, normalize, to_tensor

from hanser.models.layers import set_defaults
from hanser.models.darts.model_search_pc_darts import Network
from hanser.models.darts.genotypes import set_primitives
from hanser.train.nas.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

(x_train, y_train), (x_test, y_test) = load_cifar10_tfds()
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test = x_test[:100], y_test[:100]

from hanser.tpu import get_colab_tpu

strategy = get_colab_tpu()


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    label = tf.one_hot(label, 10)
    # image = tf.cast(image, tf.bfloat16)
    # if training:
    # image = cutout(image, 16)

    return image, label


train_portion = 0.5
n_train = len(x_train)
n_search = int(n_train * (1 - train_portion))
n_train = n_train - n_search
n_test = len(x_test)

x_search, y_search = x_train[n_search:], y_train[n_search:]
x_train, y_train = x_train[:n_train], y_train[:n_train]

mul = 1
batch_size = 16 * mul
eval_batch_size = batch_size * 2

steps_per_epoch = n_train // batch_size
test_steps = math.ceil(n_test / eval_batch_size)

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_search = tf.data.Dataset.from_tensor_slices((x_search, y_search))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds_train, batch_size, transform(training=True), training=True, buffer_size=10000)
ds_search = prepare(ds_search, batch_size, transform(training=True), training=True, buffer_size=10000)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_policy(policy)

if strategy:
    tf.distribute.experimental_set_strategy(strategy)
    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

set_defaults({
    'bn': {
        'affine': False,
        'track_running_stats': False,
    },
})
set_primitives('darts')

k = 4
# model = Network(16, 8, 4, 4, 3, k, 10)
model = Network(4, 5, 4, 4, 3, k, 10)

input_shape = (32, 32, 3)
model.build((None, *input_shape))

criterion = CrossEntropy()

base_lr = 0.1
epochs = 50

optimizer_arch = Adam(6e-4, beta_1=0.5)

lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer_model = SGD(lr_shcedule, momentum=0.9, nesterov=True)

metrics = [
    M.Mean(name='loss'), M.CategoricalAccuracy(name='acc')]
test_metrics = [
    M.CategoricalCrossentropy(name='loss', from_logits=True), M.CategoricalAccuracy(name='acc')]

trainer = Trainer(model, criterion, optimizer_arch, optimizer_model,
                  metrics, test_metrics, 5.0 / 8, 1e-3, 3e-4)


class PrintGenotype(Callback):

    def on_epoch_begin(self, epoch, logs=None):
        p = """Genotype(
    normal=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], normal_concat=[2, 3, 4, 5],
    reduce=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], reduce_concat=[2, 3, 4, 5],
)"""
        g = model.genotype()
        print(p % (tuple(g.normal) + tuple(g.reduce)))


from hanser.io import time_now
print(time_now())
trainer.fit(epochs, ds_train, ds_search, steps_per_epoch, ds_test, test_steps,
            val_freq=5, epochs_model_only=15, callbacks=[PrintGenotype()])
