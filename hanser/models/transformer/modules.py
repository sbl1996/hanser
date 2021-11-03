import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation
from hanser.models.modules import GELU


def safe_softmax(logits, axis):
    dtype = logits.dtype
    if dtype in [tf.float16, tf.bfloat16]:
        logits = tf.cast(logits, tf.float32)
        weights = tf.nn.softmax(logits, axis=axis)
        weights = tf.cast(weights, dtype)
    else:
        weights = tf.nn.softmax(logits, axis=axis)
    return weights


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = safe_softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, drop_rate, final_dense=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.drop_rate = drop_rate
        self.final_dense = final_dense

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model) if final_dense else None
        self.dropout = Dropout(drop_rate) if drop_rate else None

    def split_heads(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        output = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        if self.dense:
            output = self.dense(output)
        if self.dropout:
            output = self.dropout(output)
        return output, attention_weights

    def get_config(self):
        return {
            **super().get_config(),
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'drop_rate': self.drop_rate,
            'final_dense': self.final_dense,
        }


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(seq_len, d_model):
    angle_rads = get_angles(np.arange(seq_len)[:, None], np.arange(d_model)[None, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)


def FFN(d_model, dff, drop_rate=0, activation='gelu'):

    layers = [
        Dense(dff),
        GELU() if activation == 'gelu' else Activation('relu'),
        Dense(d_model),
    ]
    if drop_rate:
        layers.insert(2, Dropout(drop_rate))
        layers.insert(4, Dropout(drop_rate))
    return Sequential(layers)
