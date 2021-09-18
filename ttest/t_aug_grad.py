import math

from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.datasets.cifar import make_cifar10_dataset
from hanser.transform import random_crop, normalize, to_tensor, cutout

from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.datasets import prepare

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Layer
from tensorflow.keras.initializers import Constant

from hanser.train.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from hanser.models.layers import Conv2d, Linear
from hanser.ops import gumbel_softmax, sample_relaxed_bernoulli
from hanser.datasets.classification.cifar import load_cifar10
from hanser.datasets.classification.numpy import subsample


def transform(images, transforms):
    transform_or_transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    output_shape = tf.shape(images)[1:3]
    output_shape = tf.convert_to_tensor(output_shape, tf.int32)

    if len(transform_or_transforms.shape) == 1:
        transforms = transform_or_transforms[None]

    output = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=output_shape,
        interpolation='BILINEAR',
        fill_value=0.
    )
    return output


def shear_x(images, magnitude):
    return transform(
        images, [1., magnitude, 0., 0., 1., 0., 0., 0.])


def translate_x(images, pixels):
    return transform(
        images, [1, 0, -pixels, 0, 1, 0, 0, 0])


def translate_y(images, pixels):
    return transform(
        images, [1, 0, 0, 0, 1, -pixels, 0, 0])


class ShearX(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def _magnitude_to_arg(self, magnitude):
        return (self.max_val - self.min_val) * magnitude + self.min_val

    def call(self, x, magnitude):
        x = shear_x(x, self._magnitude_to_arg(magnitude))
        return x


class TranslateX(Layer):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def _magnitude_to_arg(self, magnitude):
        return (self.max_val - self.min_val) * magnitude + self.min_val

    def call(self, x, magnitude):
        x = translate_x(x, self._magnitude_to_arg(magnitude))
        return x


class DifferentiableAugmentation(Layer):

    def __init__(self, num_ops=2):
        super().__init__()
        self.sub_policies = [
            ShearX(-0.3, 0.3),
            TranslateX(-10, 10),
        ]
        num_sub_policies = len(self.sub_policies)
        self.sp_weights = self.add_weight(
            name="sp_weights", shape=(num_ops, num_sub_policies), dtype=tf.float32, trainable=True,
            initializer=Constant(1e-2), experimental_autocast=False)
        self.sp_probs = self.add_weight(
            name="sp_probs", shape=(num_ops, num_sub_policies), dtype=tf.float32, trainable=True,
            initializer=Constant(0.5), experimental_autocast=False)
        self.sp_magnitudes = self.add_weight(
            name="sp_magnitudes", shape=(num_ops, num_sub_policies), dtype=tf.float32, trainable=True,
            initializer=Constant(0.5), experimental_autocast=False)

        self.num_ops = num_ops

    def call(self, x, training=None):
        if not training:
            return x
        # x: (N, H, W, 3), float32, 0-1
        for i in range(self.num_ops):
            hardwts = gumbel_softmax(self.sp_weights[i], tau=1.0, hard=True)
            xs = [
                tf.cond(hardwts[j] == 1,
                        lambda: apply_bernoulli(
                            x, self.sub_policies[j], self.sp_probs[i, j], self.sp_magnitudes[i, j]) * hardwts[j],
                        lambda: hardwts[j])
                for j in range(len(self.sub_policies))
            ]
            x = sum(xs)
        return x


def apply_bernoulli(x, op, prob, magnitude):
    hardwts = sample_relaxed_bernoulli(prob, temperature=1.0, hard=True)
    x = x + magnitude - tf.stop_gradient(magnitude)
    x = tf.cond(hardwts == 1, lambda: op(x, magnitude) * hardwts, lambda: x * (1 - hardwts))
    return x


class ConvNet(Model):

    def __init__(self):
        super().__init__()
        self.stem = Conv2d(3, 8, kernel_size=3)

        self.flatten = Flatten()
        self.fc = Linear(8, 10)

        self.da = DifferentiableAugmentation(2)

    def call(self, x):
        x = self.da(x)

        x = self.stem(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.flatten(x)
        x = self.fc(x)
        return x


@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    label = tf.one_hot(label, 10)
    return image, label


batch_size = 64
eval_batch_size = 64

(x_train, y_train), (x_test, y_test) = load_cifar10()

x_train, y_train = subsample(x_train, y_train, ratio=0.01)
x_test, y_test = subsample(x_test, y_test, ratio=0.01)

n_val = len(x_train) // 2
x_val, y_val = x_train[n_val:], y_train[n_val:]
x_train, y_train = x_train[:n_val], y_train[:n_val]

n_train, n_test = len(x_train), len(x_test)
steps_per_epoch = n_train // batch_size

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

ds_train = prepare(ds_train, batch_size, transform(training=True), training=True,
                   buffer_size=n_train, prefetch=False)
ds_val = prepare(ds_val, batch_size, transform(training=False), training=True,
                 buffer_size=n_val, prefetch=False)
ds_trainval = tf.data.Dataset.zip((ds_train, ds_val))
ds_trainval = ds_trainval.prefetch(tf.data.experimental.AUTOTUNE)

model = ConvNet()
model.build((None, 32, 32, 3))

criterion = CrossEntropy()

base_lr = 0.1
epochs = 100
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer_model = SGD(1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
optimizer_arch = Adam(1e-3, beta_1=0.5)

train_it = iter(ds_trainval)
for i in range(10):
    # (input, target), (input_search, target_search) = next(train_it)
    model_vars = model.trainable_variables[:-3]
    arch_vars = model.trainable_variables[-3:]

    x = tf.random.uniform((4, 32, 32, 3), 0, 1, dtype=tf.float32)
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        loss = tf.reduce_mean(p)
    grads = tape.gradient(loss, model_vars)
    optimizer_model.apply_gradients(zip(grads, model_vars))

    x = tf.random.uniform((4, 32, 32, 3), 0, 1, dtype=tf.float32)
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        loss = tf.reduce_mean(p)
    grads = tape.gradient(loss, arch_vars)
    optimizer_arch.apply_gradients(zip(grads, arch_vars))