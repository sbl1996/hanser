import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d

class MixedOp(Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ops = [
            Conv2d(in_channels, out_channels, 1, bn=True, act='default'),
            Conv2d(in_channels, out_channels, 3, bn=True, act='default')
        ]

    def call(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class LeNet(Model):

    def __init__(self):
        super().__init__()
        self.op1 = MixedOp(3, 4)
        self.op2 = MixedOp(4, 5)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=[2], dtype=tf.float32, initializer=tf.keras.initializers.RandomNormal())

    def call(self, x):
        x = self.op1(x, self.alpha)
        x = self.op2(x, self.alpha)
        return x


class Trainer(Model):

    def train_step(self, data):
        (x, x_search), (y, y_search) = data

        with tf.GradientTape() as tape:
            y_pred = self(x_search, training=True)
            loss = self.compiled_loss(
                y_search, y_pred, regularization_losses=self.losses)

        trainable_variables = [ v for v in self.trainable_variables if 'alpha' in v.name]
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_variables = [v for v in self.trainable_variables if 'alpha' not in v.name]
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
