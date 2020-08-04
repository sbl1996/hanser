import tensorflow as tf
from tensorflow.keras.layers import Layer

class Linear(Layer):

    def __init__(self):
        super().__init__()
        self.w1 = self.add_weight('alpha0', (2, 3), tf.float32, 'random_normal', trainable=True) * 1e-3

    def call(self, inputs):
        return inputs * self.w1

l = Linear()