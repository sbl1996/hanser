from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.tpu import setup
from hanser.datasets.cifar import make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor, cutout, mixup, random_apply
from hanser.transform.autoaugment import autoaugment

from hanser.train.optimizers import SGD
from hanser.models.cifar.preactresnet import ResNet
from hanser.train.cls import SuperLearner
from hanser.train.callbacks import EMA, OptunaReportIntermediateResult
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

import optuna

def objective(trial: optuna.Trial):

    cutout_prob = trial.suggest_float("cutout_prob", 0, 1.0, step=0.1)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0, 0.5, step=0.1)
    label_smoothing = trial.suggest_uniform("label_smoothing", 0, 0.2)
    base_lr = trial.suggest_float("base_lr", 0.01, 0.2, step=0.01)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    ema = trial.suggest_categorical("ema", ["true", "false"])
    ema_decay = trial.suggest_loguniform("ema_decay", 0.995, 0.9999) if ema == 'true' else None

    @curry
    def transform(image, label, training):

        if training:
            image = random_crop(image, (32, 32), (4, 4))
            image = tf.image.random_flip_left_right(image)
            image = autoaugment(image, "CIFAR10")

        image, label = to_tensor(image, label)
        image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

        if training:
            image = random_apply(cutout(length=16), cutout_prob, image)

        label = tf.one_hot(label, 100)

        return image, label

    def zip_transform(data1, data2):
        return mixup(data1, data2, alpha=mixup_alpha)

    batch_size = 128
    eval_batch_size = 2048

    ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
        batch_size, eval_batch_size, transform, zip_transform)
    ds_train, ds_test = setup([ds_train, ds_test], fp16=True)

    model = ResNet(depth=16, k=8, num_classes=100)
    model.build((None, 32, 32, 3))
    model.summary()

    criterion = CrossEntropy(label_smoothing=label_smoothing)

    epochs = 50
    lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
    optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=weight_decay, nesterov=True)
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
        work_dir=f"./CIFAR100-NNI", multiple_steps=True)


    callbacks = [OptunaReportIntermediateResult('acc', trial)]
    if ema == 'true':
        callbacks.append(EMA(ema_decay))

    learner.fit(ds_train, epochs, ds_test, val_freq=1,
                       steps_per_epoch=steps_per_epoch, val_steps=test_steps,
                       callbacks=callbacks)

    return learner.metric_history.get_metric('acc', "eval")[-1]