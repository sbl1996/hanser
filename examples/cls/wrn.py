import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
import tensorflow.keras.mixed_precision.experimental as mixed_precision

import tensorflow_addons as tfa

from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10
from hanser.transform import random_crop, cutout, normalize, to_tensor
from hanser.transform.autoaugment import autoaugment

from hanser.models.cifar.preactresnet import ResNet
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

    if training:
        image = cutout(image, 16)

    # image = tf.cast(image, tf.bfloat16)
    label = tf.one_hot(label, 10)

    return image, label


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

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=50000)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

strategy = get_colab_tpu()
if strategy:
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    tf.distribute.experimental_set_strategy(strategy)

    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

# set_defaults({
#     'weight_decay': 1e-4,
# })
model = ResNet(28, 10, 0.3)
model.build((None, 32, 32, 3))

criterion = CrossEntropy()

base_lr = 0.006
epochs = 250
lr_schedule = CosineLR(base_lr * math.sqrt(mul), steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=5)
# optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True)
optimizer = tfa.optimizers.LAMB(lr_schedule, beta_1=0.9, beta_2=0.95)
metrics = [
    Mean(name='loss'), Accuracy(name='acc')]
test_metrics = [
    Loss(name='loss', from_logits=True), Accuracy(name='acc')]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics, multiple_steps=True)

hist = trainer.fit(epochs, ds_train_dist, steps_per_epoch, ds_test_dist, test_steps, val_freq=1)