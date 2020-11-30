import math

from toolz import curry

import tensorflow as tf

from tensorflow.keras.metrics import CategoricalAccuracy as Accuracy, Mean, CategoricalCrossentropy as Loss
from tensorflow.keras.datasets.mnist import load_data as load_mnist

from tensorflow_addons.optimizers import MovingAverage

from hanser.models.mnist import LeNet5

from hanser import set_seed
from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.train.optimizers import SGD
from hanser.train.v3.cls import CNNLearner
from hanser.transform import fmix, random_crop, cutout, normalize, to_tensor, cutmix

from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

@curry
def transform(image, label, training):

    image = tf.pad(image, [(2,2), (2,2), (0,0)])
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    if training:
        image = cutout(image, 16)

    # image = tf.cast(image, tf.bfloat16)
    label = tf.one_hot(label, 10)

    return image, label

def zip_transform(data1, data2):
    return cutmix(data1, data2, 1.0)


(x_train, y_train), (x_test, y_test) = load_mnist()
x_train, x_test = x_train[:, :, :, None], x_test[:, :, :, None]
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test = x_test[:100], y_test[:100]

mul = 1
num_train_examples = len(x_train)
num_test_examples = len(x_test)
batch_size = 4 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=len(ds))
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

ds_train, ds_test = setup([ds_train, ds_test], fp16=False)

model = LeNet5()
model.build((None, 32, 32, 1))

criterion = CrossEntropy()

base_lr = 0.01
epochs = 20
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=0)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)
# optimizer = tfa.optimizers.LAMB(lr_schedule, beta_1=0.9, beta_2=0.95)
optimizer = MovingAverage(optimizer, average_decay=0.998)
train_metrics = {
    "loss": Mean(),
    "acc": Accuracy(),
}
eval_metrics = {
    'loss': Loss(from_logits=True),
    'acc': Accuracy(),
}

learner = CNNLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir="checkpoints", multiple_steps=True)

# learner.load()
hist = learner.fit(ds_train, epochs, ds_test, val_freq=1, save_freq=10)