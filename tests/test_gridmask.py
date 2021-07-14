from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.tpu import setup
from hanser.datasets.mnist import make_mnist_dataset

from hanser.transform import pad, to_tensor, normalize, cutout, mixup_or_cutmix_batch

from hanser.models.mnist import LeNet5
from hanser.losses import CrossEntropy
from hanser.train.optimizers import SGD
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.train.callbacks import Callback


@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    label = tf.one_hot(label, 10)

    return image, label


def batch_transform(image, label):
    return mixup_or_cutmix_batch(
        image, label,
        mixup_alpha=0.8, cutmix_alpha=1.0,
        prob=0.5, switch_prob=0.5,
    )

batch_size = 128
eval_batch_size = 256
ds_train, ds_test, steps_per_epoch, test_steps = \
    make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=0.01,
                       batch_transform=batch_transform)

model = LeNet5()
model.build((None, 32, 32, 1))

criterion = CrossEntropy()

epochs = 20

base_lr = 0.05
lr_shcedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}


def grid_mask_fn(inputs, rate):
    if tf.random.uniform(()) < rate:
        inputs = cutout(inputs, 16)
    return inputs


class GridMask:

    def __init__(self, p):
        self.p = p
        self.rate = tf.Variable(p, dtype=tf.float32)

    def __call__(self, inputs, target):
        inputs = grid_mask_fn(inputs, self.rate)
        return inputs, target

class GridMaskSchedule(Callback):

    def __init__(self, grid_mask):
        super().__init__()
        self.grid_mask = grid_mask

    def begin_epoch(self, state):
        epochs = state['epochs']
        epoch = self.learner.epoch
        rate = (epoch + 1) / epochs * self.grid_mask.p
        self.grid_mask.rate.assign(rate)

grid_mask = GridMask(0.5)

learner = SuperLearner(
    model, criterion, optimizer, batch_transform=grid_mask,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./MNIST")

learner.fit(ds_train, epochs, ds_test, val_freq=2,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            callbacks=[GridMaskSchedule(grid_mask)])