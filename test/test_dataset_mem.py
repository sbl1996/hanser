import os

from toolz import curry

import tensorflow as tf
from hanser.datasets.cifar import make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor, cutout, mixup, mixup_batch
from hanser.transform.autoaugment import autoaugment

WORKER_ID = os.environ.get("WORKER_ID", 0)


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    # if training:
    #     image = cutout(image, 16)

    label = tf.one_hot(label, 100)

    return image, label


def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2)

def batch_transform(image, label):
    return mixup_batch(image, label, 0.2)

batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform, batch_transform=batch_transform)

train_it = iter(ds_train)
x, y = next(train_it)

# 543.2M
# mixup zip 548.8
# mixup batch 552.3