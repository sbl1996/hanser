import math

import numpy as np
from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model

from hanser.models.cifar.pyramidnest import PyramidNeSt
from hanser.datasets import prepare
from hanser.transform import random_crop, cutout, normalize, to_tensor
from hanser.train.callbacks import cosine_lr, LearningRateBatchScheduler


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

    # image = tf.cast(image, tf.bfloat16)
    # if training:
    # image = cutout(image, 16)

    return image, label


x_train, y_train = load_cifar10('train')
x_test, y_test = load_cifar10('test')

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

mul = 8
num_train_examples = 5000
num_test_examples = 1000
ds = ds.take(num_train_examples)
ds_test = ds_test.take(num_test_examples)
batch_size = 2 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds_train = prepare(ds, batch_size, preprocess(training=True), training=True, buffer_size=10000)
ds_test = prepare(ds_test, eval_batch_size, preprocess(training=False), training=False)


class Trainer(Model):

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # y_pred = tf.cast(y_pred, tf.float32)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        y_pred = tf.cast(y_pred, tf.float32)
        self.compiled_loss(y, y_pred)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


input_shape = (32, 32, 3)
model = PyramidNeSt(4, 12, 20, 1, 1, 10)
input = tf.keras.Input(input_shape)
trainer = Trainer(inputs=input, outputs=model(input))
trainer.summary()

criterion = SparseCategoricalCrossentropy(from_logits=True)

base_lr = 0.1
lr_shcedule = cosine_lr(base_lr=base_lr * mul, epochs=200, min_lr=1e-5,
                        warmup_epoch=5, warmup_min_lr=base_lr)
# optimizer = tfa.optimizers.SGDW(2e-4, base_lr * mul, momentum=0.9, nesterov=True)
optimizer = SGD(base_lr * mul, momentum=0.9, nesterov=True)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
trainer.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

callbacks = [LearningRateBatchScheduler(lr_shcedule, steps_per_epoch)]

trainer.fit(ds_train, epochs=200, steps_per_epoch=steps_per_epoch,
            validation_data=ds_test, validation_steps=test_steps,
            validation_freq=2, verbose=1, callbacks=callbacks)

it = iter(ds_train)
x, y = next(it)
trainer.train_on_batch(x, y)
