import math

import numpy as np
from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean, CategoricalAccuracy

from hanser.models.functional.cifar.pyramidnet import PyramidNet
from hanser.models.functional.layers import DEFAULTS
from hanser.datasets import prepare
from hanser.train.lr_schedule import CosineLR
from hanser.transform import random_crop, cutout, normalize, to_tensor, mixup, random_apply2
from hanser.tpu import get_colab_tpu
from hanser.train.trainer import Trainer

import tensorflow.keras.mixed_precision.experimental as mixed_precision

from hanser.transform.autoaugment import autoaugment


def load_cifar10(split):
    ds = tfds.as_numpy(tfds.load('cifar10', split=split, data_dir='./cifar10'))
    x = []
    y = []
    for d in ds:
        x.append(d['image'])
        y.append(d['label'])
    x = np.stack(x)
    y = np.stack(y)
    return x, y


x_train, y_train = load_cifar10('train')
x_test, y_test = load_cifar10('test')

strategy = get_colab_tpu()


@curry
def preprocess(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    # image = tf.cast(image, tf.bfloat16)
    if training:
        image = cutout(image, 16)

    return image, label

def batch_preprocess(image, label):
    image, label = random_apply2(mixup(beta=1.0), 1.0, image, label)

    return image, label

mul = 8
num_train_examples, num_test_examples = 50000, 10000
batch_size = 128 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds = strategy.experimental_make_numpy_dataset((x_train, y_train))
ds_test = strategy.experimental_make_numpy_dataset((x_test, y_test))

ds_train = prepare(ds, preprocess(training=True), batch_size, training=True, buffer_size=10000,
                   batch_preprocess=batch_preprocess)
ds_test = prepare(ds_test, preprocess(training=False), eval_batch_size, training=False)

tf.distribute.experimental_set_strategy(strategy)
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)

ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

DEFAULTS['weight_decay'] = 1e-4
input_shape = (32, 32, 3)
model = PyramidNet(input_shape, 32, 480 - 32, 56, 16, 1, 0.2, 10)
model.summary()

criterion = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1, reduction='none')
# criterion = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

base_lr = 0.1
epochs = 600
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=1e-5, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
metrics = [
    Mean(name='loss'), CategoricalAccuracy(name='acc')]
test_metrics = [
    Mean(name='loss'), CategoricalAccuracy(name='acc')]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics)

trainer.fit(epochs, ds_train_dist, steps_per_epoch, ds_test_dist, test_steps, val_freq=5)
