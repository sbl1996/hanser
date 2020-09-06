import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.callbacks import Callback

from hanser import set_seed
from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets import load_mnist
from hanser.transform import pad, to_tensor, normalize

from hanser.models.mnist import LeNet5
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

import nni

@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    label = tf.one_hot(label, 10)
    # image = tf.cast(image, tf.bfloat16)

    return image, label


def main(params):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = x_train[:, :, :, None], x_test[:, :, :, None]
    x_train, y_train = x_train[:5000], y_train[:5000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    mul = 1
    num_train_examples = len(x_train)
    num_test_examples = len(x_test)
    batch_size = params['batch_size'] * mul
    eval_batch_size = batch_size * 2
    steps_per_epoch = num_train_examples // batch_size
    test_steps = math.ceil(num_test_examples / eval_batch_size)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=10000)
    ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

    set_defaults({
        'weight_decay': params["weight_decay"],
    })
    model = LeNet5()
    model.build((None, 32, 32, 1))

    criterion = CrossEntropy()

    epochs = int(params['epochs'])

    base_lr = params["learning_rate"]
    lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs, min_lr=0,
                           warmup_min_lr=base_lr, warmup_epoch=10)
    optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
    metrics = [
        Mean(name='loss'), Accuracy(name='acc')]
    test_metrics = [
        Loss(name='loss', from_logits=True), Accuracy(name='acc')]

    trainer = Trainer(model, criterion, optimizer, metrics, test_metrics)

    class ReportIntermediates(Callback):

        def on_epoch_end(self, epoch, logs=None):
            val_acc = trainer.metric_history.get_metric("acc", "Valid", epoch, epoch)
            if val_acc:
                nni.report_intermediate_result(val_acc)

    hist = trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps,
                       val_freq=2, valid_after=10, callbacks=[ReportIntermediates()])
    nni.report_final_result(hist['acc'][-1])


if __name__ == '__main__':
    set_seed(42)
    params = {
        'batch_size': 32,
    }

    # fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)

    print('Hyper-parameters: %s' % params)
    main(params)
