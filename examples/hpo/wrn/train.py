import math

from toolz import curry

import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision

import tensorflow_addons as tfa

from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10, load_cifar100
from hanser.transform import random_crop, cutout, normalize, to_tensor

from hanser.models.cifar.preactresnet import ResNet
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

import nni

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])

    if training:
        image = cutout(image, 16)

    image = tf.cast(image, tf.bfloat16)
    label = tf.one_hot(label, 10)

    return image, label

def main(params):

    (x_train, y_train), (x_test, y_test) = load_cifar10()

    mul = 8
    num_train_examples = len(x_train)
    num_test_examples = len(x_test)
    batch_size = 128 * mul
    eval_batch_size = batch_size * (16 // mul)
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

    base_lr = params['learning_rate']
    epochs = 55
    lr_schedule = CosineLR(base_lr * math.sqrt(mul), steps_per_epoch, epochs=epochs,
                           min_lr=0, warmup_min_lr=base_lr, warmup_epoch=5)
    # optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True)
    optimizer = tfa.optimizers.LAMB(lr_schedule, beta_2=params['beta2'], weight_decay_rate=params['weight_decay'])
    metrics = [
        Mean(name='loss'), Accuracy(name='acc')]
    test_metrics = [
        Loss(name='loss', from_logits=True), Accuracy(name='acc')]

    trainer = Trainer(model, criterion, optimizer, metrics, test_metrics, multiple_steps=True)

    class ReportIntermediates(Callback):

        def on_epoch_end(self, epoch, logs=None):
            val_acc = trainer.metric_history.get_metric("acc", "Valid", epoch, epoch)
            if val_acc:
                nni.report_intermediate_result(val_acc)

    hist = trainer.fit(epochs, ds_train_dist, steps_per_epoch, ds_test_dist, test_steps,
                       val_freq=1, callbacks=[ReportIntermediates()])
    nni.report_final_result(np.max(hist['acc']))


if __name__ == '__main__':
    params = {}

    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)

    print('Hyper-parameters: %s' % params)
    main(params)
