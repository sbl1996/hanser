from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.tpu import setup
from hanser.datasets.cifar import make_cifar100_dataset
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

batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform)
ds_train, ds_test = setup([ds_train, ds_test], fp16=True)

model = ResNet(depth=16, k=8, num_classes=100)
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy(label_smoothing=0)

base_lr = 0.1
epochs = 200
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
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