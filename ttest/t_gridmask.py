from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy
import tensorflow_addons as tfa

from hanser.tpu import setup
from hanser.datasets.mnist import make_mnist_dataset

from hanser.transform import pad, to_tensor, normalize, cutout, mixup_or_cutmix_batch, image_dimensions

from hanser.models.mnist import LeNet5
from hanser.losses import CrossEntropy
from hanser.train.optimizers import SGD
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.train.callbacks import Callback


def grid_mask_batch(images, d_min=24, d_max=32, rotate=0, ratio=0.4, p=0.8):
    # this is a must to rotate batched images, because xla does not support it currently
    tf.config.set_soft_device_placement(True)
    n, h, w, c = image_dimensions(images, 4)
    hh = tf.cast(tf.math.sqrt(
        tf.cast(h * h + w * w, tf.float32)), tf.int32)  # to ensure coverage after rotation

    d = tf.cast(tf.random.uniform((n, 1), d_min, d_max + 1, dtype=tf.int32), tf.float32)  # the length of a unit's edge
    d_x = tf.cast(tf.math.ceil(tf.random.uniform((n, 1), tf.zeros_like(d), d)), dtype=tf.int32)  # bias
    d_y = tf.cast(tf.math.ceil(tf.random.uniform((n, 1), tf.zeros_like(d), d)), dtype=tf.int32)
    l = tf.cast(d * ratio, tf.int32)
    d = tf.cast(d, tf.int32)

    # generate masks
    idx = tf.repeat(tf.expand_dims(tf.range(hh), 0), repeats=n, axis=0)
    masks_x = tf.cast(tf.math.logical_and(l <= (idx - d_x) % d, (idx - d_x) % d < d), tf.float32)
    masks_y = tf.cast(tf.math.logical_and(l <= (idx - d_y) % d, (idx - d_y) % d < d), tf.float32)
    masks = tf.matmul(tf.expand_dims(masks_y, -1), tf.expand_dims(masks_x, 1))
    masks = tf.cast(masks == 0, tf.float32)

    masks = tf.repeat(tf.expand_dims(masks, -1), repeats=c, axis=-1)

    # rotate
    angles = tf.random.uniform((n,), 0, rotate, dtype=tf.float32)
    masks = tfa.image.rotate(masks, angles)

    # get the center part
    masks = masks[:, (hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w, :]

    # probability
    masks = tf.where(tf.random.uniform((n, 1, 1, 1)) < p, masks, tf.ones_like(masks))

    images *= tf.cast(masks, images.dtype)
    print(images.shape)
    return images


class GridMask:
    def __init__(self, d_min=24, d_max=32, rotate=0, ratio=0.4, p=0.8):
        self.d_min = d_min
        self.d_max = d_max
        self.rotate = rotate
        self.ratio = ratio
        self.init_p = p
        self.p = None

    def __call__(self, images, labels):
        if self.p is None:
            self.p = tf.Variable(self.init_p, tf.float32)
        images = grid_mask_batch(images, self.d_min, self.d_max, self.rotate, self.ratio, self.p)
        return images, labels


class GridMaskSchedule(Callback):
    def __init__(self, gm, till=0):
        super().__init__()
        self.gm = gm
        self.till = till  # when to stop ascending, till==0 means no ascending

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        if epoch >= self.till:
            return
        elif epoch == 0:
            rate = 1 / self.till * self.gm.p
        else:
            rate = (epoch + 1) / epoch * self.gm.p
        self.gm.p.assign(rate)


@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    label = tf.one_hot(label, 10)

    return image, label

grid_mask = GridMask()

batch_size = 128
eval_batch_size = 256
ds_train, ds_test, steps_per_epoch, test_steps = \
    make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=0.01,
                       batch_transform=grid_mask)

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

learner.fit(ds_train, epochs, ds_test, val_freq=2,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            callbacks=[GridMaskSchedule(grid_mask, 15)])