import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

class DropBlock(Layer):

    def __init__(self, keep_prob, block_size, gamma_mul=1., **kwargs):
        super().__init__(**kwargs)
        assert block_size % 2 == 1
        self.block_size = block_size
        self.gamma_mul = gamma_mul

        self.keep_prob = self.add_weight(
            name="keep_prob", shape=(), dtype=tf.float32,
            initializer=initializers.Constant(keep_prob), trainable=False,
        )

    def call(self, inputs, training=None):
        if training:
            br = (self.block_size - 1) // 2
            tl = (self.block_size - 1) - br

            n = tf.shape(inputs)[0]
            h, w, c = inputs.shape[1:]
            sampling_mask_shape = [n, h - self.block_size + 1, w - self.block_size + 1, 1]
            pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]

            ratio = (w * h) / (self.block_size ** 2) / ((w - self.block_size + 1) * (h - self.block_size + 1))
            gamma = (1. - self.keep_prob) * ratio * self.gamma_mul
            mask = tf.cast(
                tf.random.uniform(sampling_mask_shape) < gamma, tf.float32)
            mask = tf.pad(mask, pad_shape)

            kernel_size = self.block_size // 2 + 1
            mask = tf.nn.max_pool2d(mask, kernel_size, strides=1, padding='SAME')

            mask = 1. - mask
            mask_reduce_sum = tf.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
            normalize_factor = tf.cast(h * w, dtype=tf.float32) / (mask_reduce_sum + 1e-8)

            ret = inputs * tf.cast(mask, inputs.dtype) * tf.cast(normalize_factor, inputs.dtype)
            return ret
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'keep_prob': self.keep_prob,
            "block_size": self.block_size,
            "gamma_mul": self.gamma_mul,
        }