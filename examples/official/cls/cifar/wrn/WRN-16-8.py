from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy
from tensorflow_addons.optimizers import MovingAverage

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.classification.cifar import make_cifar100_dataset

from hanser.transform import random_crop, normalize, to_tensor, cutout, mixup
from hanser.transform.autoaugment import autoaugment

from hanser.train.optimizers import SGD
from hanser.models.cifar.preactresnet import ResNet
from hanser.train.cls import SuperLearner
from hanser.train.callbacks import EMA
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy


@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    if training:
        image = cutout(image, 16)

    label = tf.one_hot(label, 100)

    return image, label


def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2)


batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform, zip_transform)

setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

model = ResNet(depth=16, k=8, num_classes=100)
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy()

base_lr = 0.1
epochs = 300
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer = MovingAverage(optimizer, average_decay=0.9999)
train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir="./drive/MyDrive/models/CIFAR100")

learner.fit(ds_train, epochs, ds_test, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            callbacks=[EMA()])
