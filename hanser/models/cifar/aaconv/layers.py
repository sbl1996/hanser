import tensorflow as tf
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Pool2d


def split_heads_2d(x, n_heads):
    B = tf.shape(x)[0]
    H, W, d = x.shape[1:]
    split = tf.reshape(x, [B, H, W, n_heads, d // n_heads])
    return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(x):
    x = tf.transpose(x, [0, 2, 3, 1, 4])
    B = tf.shape(x)[0]
    H, W, n_heads, d = x.shape[1:]
    return tf.reshape(x, [B, H, W, n_heads * d])


def flatten_hw(x):
    B = tf.shape(x)[0]
    n_heads, H, W, d = x.shape[1:]
    return tf.reshape(x, [B, n_heads, H * W, d])


def split_hw(x, H, W):
    # (B, n_heads, H * W, d)
    B = tf.shape(x)[0]
    n_heads, d = x.shape[1], x.shape[3]
    return tf.reshape(x, [B, n_heads, H, W, d])


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
    return output


class MultiHeadAttention(Layer):

    def __init__(self, c_in, n_heads, dk, dv, stride=1):
        super().__init__()
        assert dv % n_heads == 0
        self.n_heads = n_heads
        self.stride = stride
        if self.stride == 2:
            self.downsample = Pool2d(3, 2, type='avg')

        dh_k = max(dk // n_heads, 20)
        self.Q = Conv2d(c_in, dh_k * n_heads, 1)
        self.K = Conv2d(c_in, dh_k * n_heads, 1)
        self.V = Conv2d(c_in, dv, 1)

        self.project = Conv2d(dv, dv, 1)

    def call(self, x):
        """
            x: feature map of shape (N, H, W, c_in)
            :return: (N, H, W, d_v) or (N, H // 2, W // 2, d_v)
        """
        if self.stride == 2:
            x = self.downsample(x)

        H, W = x.shape[1:3]

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = flatten_hw(split_heads_2d(q, self.n_heads))
        k = flatten_hw(split_heads_2d(k, self.n_heads))
        v = flatten_hw(split_heads_2d(v, self.n_heads))

        out = scaled_dot_product_attention(q, k, v)
        out = split_hw(out, H, W)
        out = combine_heads_2d(out)

        out = self.project(out)
        return out


class AAConv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, n_heads, dk, dv):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels - dv, kernel_size, stride=stride)
        self.att = MultiHeadAttention(in_channels, n_heads, dk, dv, stride=stride)

    def call(self, x):
        x1 = self.conv(x)
        x2 = self.att(x)
        return tf.concat([x1, x2], axis=-1)