import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics as M
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from hanser import set_seed
from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10
from hanser.transform import random_crop, cutout, normalize, to_tensor, random_apply2, mixup, cutmix
from hanser.transform.autoaugment import autoaugment

from hanser.models.cifar.pyramidnext import PyramidNeXt
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    label = tf.one_hot(label, 10)
    # image = tf.cast(image, tf.bfloat16)
    # if training:
    #     image = cutout(image, 16)

    return image, label


def zip_transform(data1, data2):
    return tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: cutmix(data1, data2, beta=1.0),
        lambda: data1,
    )

set_seed(42)

(x_train, y_train), (x_test, y_test) = load_cifar10()
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test = x_test[:100], y_test[:100]

mul = 1
num_train_examples = len(x_train)
num_test_examples = len(x_test)
batch_size = 32 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=10000,
                   zip_transform=zip_transform)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

strategy = get_colab_tpu()
if strategy:
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    tf.distribute.experimental_set_strategy(strategy)

    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

set_defaults({
    'weight_decay': 1e-4
})
drop_path = 0.2
model = PyramidNeXt(4, 16 - 4, 20, 1, True, drop_path=drop_path, use_aux_head=True)
model.build((None, 32, 32, 3))

criterion = CrossEntropy(auxiliary_weight=0.4)

base_lr = 0.1
epochs = 600
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=10)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
metrics = [
    M.Mean(name='loss'), M.CategoricalAccuracy(name='acc')]
test_metrics = [
    M.CategoricalCrossentropy(name='loss', from_logits=True), M.CategoricalAccuracy(name='acc')]
metric_transform = lambda x: x[0]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics,
                  metric_transform=metric_transform, multiple_steps=True)


class DropPathRateSchedule(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        rate = epoch / epochs * drop_path
        for l in model.submodules:
            if 'drop' in l.name:
                l.rate = rate


# trainer.fit(epochs, ds_train_dist, steps_per_epoch, ds_test_dist, test_steps, val_freq=5)
trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps, val_freq=5)