import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10_tfds
from hanser.transform import random_crop, cutout, normalize, to_tensor
from hanser.transform.autoaugment import autoaugment

from hanser.models.cifar.pyramidnext import PyramidNeXt
from hanser.models.layers import set_default
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

ds_train = prepare(ds, transform(training=True), batch_size, training=True, buffer_size=10000)
ds_test = prepare(ds_test, transform(training=False), eval_batch_size, training=False)

# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_policy(policy)

if strategy:
    tf.distribute.experimental_set_strategy(strategy)
    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

set_default(['weight_decay'], 1e-4)
drop_path = 0.2
# model = PyramidNeXt(32, 480-32, 56, 16, True, drop_path, True, 10)
model = PyramidNeXt(4, 16-4, 20, 1, True, drop_path, True, 10)

criterion = CrossEntropy(auxiliary_weight=0.4)

base_lr = 0.01
epochs = 100
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=1e-5, warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
metrics = [
    Mean(name='loss'), Accuracy(name='acc')]
test_metrics = [
    Loss(name='loss', from_logits=True), Accuracy(name='acc')]
metric_transform = lambda x: x[0]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics, metric_transform=metric_transform)

trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps, val_freq=5)
