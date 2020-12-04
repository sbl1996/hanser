import math

from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar100
from hanser.transform import random_crop, normalize, to_tensor

from hanser.train.optimizers import SGD
from hanser.models.cifar.preactresnet import ResNet
from hanser.train.v3.cls import CNNLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    label = tf.one_hot(label, 100)

    return image, label

(x_train, y_train), (x_test, y_test) = load_cifar100()

mul = 1
n_train, n_test = len(x_train), len(x_test)
batch_size = 128 * mul
eval_batch_size = batch_size * (16 // mul)
steps_per_epoch = n_train // batch_size
test_steps = math.ceil(n_test / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds, batch_size, transform=transform(training=True),
                   training=True, buffer_size=len(x_train))
ds_test = prepare(ds_test, eval_batch_size, transform=transform(training=False), training=False)

ds_train, ds_test = setup([ds_train, ds_test], fp16=True)

model = ResNet(depth=16, k=8, num_classes=100)
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy(label_smoothing=0)

base_lr = 0.1
epochs = 200
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=5e-4, nesterov=True)
train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = CNNLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir="./drive/My Drive/models/CIFAR100-1", multiple_steps=True)

hist = learner.fit(ds_train, epochs, ds_test, val_freq=1,
                   steps_per_epoch=steps_per_epoch, val_steps=test_steps)