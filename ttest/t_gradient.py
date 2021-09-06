import os

import numpy as np
from toolz import curry
from hanser.datasets.mnist import make_mnist_dataset

import tensorflow as tf

from tensorflow.keras.layers import Flatten, Layer
from hanser.transform import to_tensor, normalize, pad

from hanser.models.layers import Conv2d, Linear


@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    # label = tf.one_hot(label, 10)

    return image, label


batch_size = 128
eval_batch_size = 256
ds_train, ds_test, steps_per_epoch, test_steps = \
    make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=0.01)



class Transform(tf.keras.Layer):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def call(self, x):
        x = tf.raw_ops.ImageProjectiveTransformV2(
            images=x,
            transforms=[[1., 0., 0., 0.5, 1., 0., 0., 0.]],
            output_shape=tf.shape(x)[1:3],
            interpolation="BILINEAR",
        )
        return x


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)
    return extended



class ShearX(Layer):

    def __init__(self, magnitude):
        super().__init__()
        self.magnitude = self.add_weight(
            name='magnitude', shape=(), initializer=np.array(magnitude), trainable=False)

    def call(self, x):
        x = tf.raw_ops.ImageProjectiveTransformV2(
            images=x,
            transforms=[[1., self.magnitude, 0., 0., 1., 0., 0., 0.]],
            output_shape=tf.shape(x)[1:3],
            interpolation="BILINEAR",
        )
        return x



class ConvNet(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stem = Conv2d(1, 4, kernel_size=1)

        self.flatten = Flatten()
        self.fc = Linear(4, 10)

        self.wt = self.add_weight(
            name="wt", shape=(), dtype=tf.float32, trainable=True, initializer='ones')

    def call(self, x):
        x = x * self.wt

        x = self.stem(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.flatten(x)
        x = self.fc(x)
        return x


model = ConvNet()
model.build((None, 8, 8, 1))
# optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

loss_m = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(batch):
    x, y = batch
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(y, p, from_logits=True)
        loss = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_m.update_state(per_example_loss)
    accuracy.update_state(y, p)


epochs = 200
for epoch in range(epochs):
    it = iter(ds_train)
    loss_m.reset_state()
    accuracy.reset_state()
    for step in range(steps_per_epoch):
        batch = next(it)
        train_step(batch)
    print("Epoch %d/%d" % (epoch + 1, epochs))
    print("Train - loss: %.4f, acc: %.4f" % (loss_m.result(), accuracy.result()))
    print(model.wt.numpy())