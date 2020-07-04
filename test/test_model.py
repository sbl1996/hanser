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

num_train_examples = 5000
num_test_examples = 1000
ds = ds.take(num_train_examples)
ds_test = ds_test.take(num_test_examples)
batch_size = 2
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)


ds_train = prepare(ds, preprocess(training=True), batch_size, training=True, buffer_size=10000)
ds_test = prepare(ds_test, preprocess(training=False), eval_batch_size, training=False)


class Trainer(Model):

    def train_step(self, data):
        print(data)
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

input_shape = (32, 32, 3)
model = PyramidNeSt(4, 12, 20, 1, 1, 10)
# model.build((None,) + input_shape)
# model.call(tf.keras.Input(input_shape))
input = tf.keras.Input(input_shape)
trainer = Trainer(inputs=input, outputs=model(input))
trainer.summary()

criterion = SparseCategoricalCrossentropy(from_logits=True)
optimizer = SGD(1e-2, momentum=0.9, nesterov=True)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
trainer.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

trainer.fit(ds_train, epochs=200, steps_per_epoch=steps_per_epoch,
          validation_data=ds_test, validation_steps=test_steps,
          validation_freq=10, verbose=1)


it = iter(ds_train)
x, y = next(it)
trainer.train_on_batch(x, y)