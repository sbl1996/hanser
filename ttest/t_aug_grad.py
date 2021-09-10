import tensorflow as tf
from tensorflow.keras.layers import Flatten, Layer
from hanser.models.layers import Conv2d, Linear

class Transform(Layer):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def call(self, x):
        x = tf.raw_ops.ImageProjectiveTransformV2(
            images=x,
            transforms=[self.transform],
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

        self.t1 = Transform([1., 0., 0., 0.5, 1., 0., 0., 0.])
        self.wt1 = self.add_weight(
            name="wt1", shape=(), dtype=tf.float32, trainable=True, initializer='ones')

        self.t2 = Transform([1., 0., 0., 1., 0.5, 0., 0., 0.])
        self.wt2 = self.add_weight(
            name="wt2", shape=(), dtype=tf.float32, trainable=True, initializer='ones')


    def call(self, x):
        x = self.t1(x) * self.wt1 + self.t2(x) * self.wt2

        x = self.stem(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = ConvNet()
model.build((None, 8, 8, 3))

x = tf.random.uniform((2, 8, 8, 3), 0, 1, dtype=tf.float32)
with tf.GradientTape() as tape:
    p = model(x, training=True)
    loss = tf.reduce_mean(p)

    grads = tape.gradient(loss, model.trainable_variables)

print(grads[-1], grads[-2])