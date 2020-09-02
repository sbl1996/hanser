import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics as M
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from torchvision.transforms import RandomCrop

from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10
from hanser.transform import random_crop, cutout, normalize, to_tensor

from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import Genotype
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    if training:
        image = cutout(image, 16)

    # image = tf.cast(image, tf.bfloat16)
    label = tf.one_hot(label, 10)

    return image, label

RandomCrop
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

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=10000)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

strategy = get_colab_tpu()
if strategy:
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    tf.distribute.experimental_set_strategy(strategy)

    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)


PDARTS = Genotype(
    normal=[
        ('skip_connect', 0), ('dil_conv_3x3', 1),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0), ('dil_conv_3x3', 1),
        ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5],
)

set_defaults({
    'weight_decay': 3e-4
})
drop_path = 0.2
# model = DARTS(36, 20, True, drop_path, 10, PDARTS)
model = NASNet(4, 5, True, drop_path, 10, PDARTS)
model.build((None, 32, 32, 3))
# model.call(tf.keras.layers.Input((32, 32, 3)))
# model.summary()

criterion = CrossEntropy(auxiliary_weight=0.4)
model.fit
base_lr = 0.025
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
                  grad_clip_norm=5.0, metric_transform=metric_transform,
                  multiple_steps=True)


class DropPathRateSchedule(Callback):

    def on_epoch_begin(self, epoch, logs=None):
        rate = epoch / epochs * drop_path
        for l in model.submodules:
            if 'drop' in l.name:
                l.rate = rate


trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps, val_freq=5,
            callbacks=[DropPathRateSchedule()])
