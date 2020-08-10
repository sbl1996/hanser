import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10_tfds
from hanser.transform import random_crop, cutout, normalize, to_tensor, random_apply2, mixup, cutmix

from hanser.models.cifar.general_darts import DARTS
from hanser.models.darts.genotypes import Genotype
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

(x_train, y_train), (x_test, y_test) = load_cifar10_tfds()
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test = x_test[:100], y_test[:100]

from hanser.tpu import get_colab_tpu

strategy = get_colab_tpu()


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

    label = tf.one_hot(label, 10)
    # image = tf.cast(image, tf.bfloat16)
    if training:
        image = cutout(image, 16)

    return image, label


mul = 2
num_train_examples = len(x_train)
num_test_examples = len(x_test)
batch_size = 16 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=10000)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_policy(policy)

if strategy:
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
model = DARTS(4, 5, True, drop_path, 10, PDARTS)
model.build((None, 32, 32, 3))
model.call(tf.keras.layers.Input((32, 32, 3)))
model.summary()

criterion = CrossEntropy(auxiliary_weight=0.4)

base_lr = 0.025
epochs = 600
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=1e-5, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
metrics = [
    Mean(name='loss'), Accuracy(name='acc')]
test_metrics = [
    Loss(name='loss', from_logits=True), Accuracy(name='acc')]
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
