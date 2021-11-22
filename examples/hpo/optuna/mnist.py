import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.datasets.mnist import make_mnist_dataset
from hanser.transform import pad, to_tensor, normalize
from hanser.models.mnist import LeNet5
from hanser.train.optimizers import SGD
from hanser.train.cls import SuperLearner
from hanser.train.callbacks import OptunaReportIntermediateResult
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

import optuna

def objective(trial: optuna.Trial):

    base_lr = trial.suggest_float("base_lr", 0.001, 0.05, step=0.001)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    ema = trial.suggest_categorical("ema", ["true", "false"])
    ema_decay = trial.suggest_loguniform("ema_decay", 0.99, 0.9999) if ema == 'true' else None


    @curry
    def transform(image, label, training):
        image = pad(image, 2)
        image, label = to_tensor(image, label)
        image = normalize(image, [0.1307], [0.3081])

        label = tf.one_hot(label, 10)

        return image, label


    batch_size = 128
    eval_batch_size = 256
    ds_train, ds_test, steps_per_epoch, test_steps = make_mnist_dataset(
        batch_size, eval_batch_size, transform, sub_ratio=0.01)

    model = LeNet5()
    model.build((None, 32, 32, 1))

    criterion = CrossEntropy()

    epochs = 20

    lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
    optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=weight_decay)

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
        work_dir=f"./MNIST", multiple_steps=True)
    learner._verbose = True
    callbacks = [OptunaReportIntermediateResult('acc', trial)]
    # if ema == 'true':
    #     callbacks.append(EMA(ema_decay))

    learner.fit(ds_train, epochs, ds_test, val_freq=2,
                       steps_per_epoch=steps_per_epoch, val_steps=test_steps,
                       callbacks=callbacks)

    return learner.metric_history.get_metric('acc', "eval")[-1]


from hanser.hpo.optuna import optimize_mp
study = optuna.create_study(
    direction="maximize",
    study_name="mnist1",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=5, interval_steps=2),
    storage="sqlite:///mnist1.db"
)
optimize_mp(study, objective, n_trials=10)