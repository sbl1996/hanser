from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.datasets.mnist import make_mnist_dataset

from hanser.transform import pad, to_tensor, normalize, mixup_or_cutmix_batch

from hanser.models.mnist import LeNet5
from hanser.train.optimizers import SGD
from hanser.train.cls import SuperLearner
from hanser.train.callbacks import Callback
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy


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
        mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5)

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

learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./MNIST")


learner.fit(ds_train, epochs, steps_per_epoch=steps_per_epoch)
