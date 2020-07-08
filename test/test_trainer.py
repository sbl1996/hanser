import math

import numpy as np
from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.callbacks import Callback

from hanser.models.functional.cifar.pyramidnet import PyramidNet
from hanser.datasets import prepare
from hanser.train.trainer import Trainer
from hanser.transform import random_crop, cutout, normalize, to_tensor, random_apply2, mixup, cutmix
from hanser.train.lr_schedule import CosineLR


def load_cifar10(split):
    ds = tfds.as_numpy(tfds.load('cifar10', split=split))
    x = []
    y = []
    for d in ds:
        x.append(d['image'])
        y.append(d['label'])
    x = np.stack(x)
    y = np.stack(y)
    return x, y


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

    # if training:
    # image = cutout(image, 16)

    return image, label


def batch_preprocess(image, label):
    image, label = random_apply2(cutmix(beta=1.0), 0.5, image, label)

    return image, label


x_train, y_train = load_cifar10('train')
x_test, y_test = load_cifar10('test')

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

mul = 8
num_train_examples = 500
num_test_examples = 100
ds = ds.take(num_train_examples)
ds_test = ds_test.take(num_test_examples)
batch_size = 2 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds_train = prepare(ds, preprocess(training=True), batch_size, training=True, buffer_size=10000, batch_preprocess=batch_preprocess)
ds_test = prepare(ds_test, preprocess(training=False), eval_batch_size, training=False)

input_shape = (32, 32, 3)
drop_path = 0
model = PyramidNet(input_shape, 4, 12, 20, 1, True, drop_path, 10)
# model = PyramidNet(input_shape, 32, 480-32, 56, 16, True, 0, 10)

# input_shape = (None, 32, 32, 3)
# model2 = PyramidNeSt(4, 12, 20, 1, 1, 0, 10)
# model2 = PyramidNeSt(32, 480-32, 56, 16, 1, 0, 10)
# model2.build(input_shape)
# input = tf.keras.Input(input_shape[1:])
# model2.call(input)
# model2.summary()

criterion = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1, reduction='none')
# criterion = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

base_lr = 0.01
epochs = 20
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=1e-5, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)

metrics = [
    Mean(name='loss'), Accuracy(name='acc')]
test_metrics = [
    Loss(name='loss', from_logits=True), Accuracy(name='acc')]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics)


class DropRateDecay(Callback):

    def on_epoch_begin(self, epoch, logs=None):
        rate = (epoch + 1) / epochs * drop_path
        for l in model.layers:
            if 'drop' in l.name:
                # K.set_value(l.rate, rate)
                l.rate = rate

trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps, callbacks=[DropRateDecay()])
