from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers


class DropBlock(Layer):

    def __init__(self, keep_prob, block_size, gamma_mul=1., **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.gamma_mul = gamma_mul

        self.keep_prob = self.add_weight(
            name="keep_prob", shape=(), dtype=tf.float32,
            initializer=initializers.Constant(keep_prob), trainable=False,
        )

    def call(self, x, training=None):
        if training:
            br = (self.block_size - 1) // 2
            tl = (self.block_size - 1) - br

            n = tf.shape(x)[0]
            h, w, c = x.shape[1:]
            sampling_mask_shape = [n, h - self.block_size + 1, w - self.block_size + 1, 1]
            pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]

            ratio = (w * h) / (self.block_size ** 2) / ((w - self.block_size + 1) * (h - self.block_size + 1))
            gamma = (1. - self.keep_prob) * ratio * self.gamma_mul
            mask = tf.cast(
                tf.random.uniform(sampling_mask_shape) < gamma, tf.float32)
            mask = tf.pad(mask, pad_shape)

            mask = tf.nn.max_pool2d(mask, self.block_size, strides=1, padding='SAME')
            mask = 1. - mask
            mask_reduce_sum = tf.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
            normalize_factor = tf.cast(h * w, dtype=tf.float32) / (mask_reduce_sum + 1e-8)

            x = x * tf.cast(mask, x.dtype) * tf.cast(normalize_factor, x.dtype)
            return x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'keep_prob': self.keep_prob,
            "block_size": self.block_size,
            "gamma_mul": self.gamma_mul,
        }

dp = DropBlock(0.9, block_size=14)
im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))

x = tf.cast(x, dtype=tf.float32)
# x2 = cutout2(x, 112, 'uniform')
xs = tf.stack([x for _ in range(4)], axis=0)
xs2, mask1, mask2 = dp(xs, training=True)
x2 = xs2[0]
# x2 = shear_x(xt, 1, 0)
x2 = x2.numpy()
x2 = x2.astype(np.uint8)
Image.fromarray(x2).show()

mask3 = 1 - mask1 + mask2

plt.imshow(mask3[0, :, :, 0].numpy()); plt.show()