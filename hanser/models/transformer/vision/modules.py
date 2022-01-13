import tensorflow as tf
from tensorflow.keras.layers import Layer
from hanser.models.layers import Linear, Act, Norm
from hanser.models.modules import Dropout, DropPath, Identity


def safe_softmax(logits, axis):
    dtype = logits.dtype
    if dtype in [tf.float16, tf.bfloat16]:
        logits = tf.cast(logits, tf.float32)
        weights = tf.nn.softmax(logits, axis=axis)
        weights = tf.cast(weights, dtype)
    else:
        weights = tf.nn.softmax(logits, axis=axis)
    return weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads

        self.qkv = Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop) if attn_drop > 0 else Identity()
        self.proj = Linear(d_model, d_model)
        self.proj_drop = Dropout(proj_drop) if proj_drop > 0 else Identity()

    def call(self, x):
        B = tf.shape(x)[0]
        N, C = x.shape[1:]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, self.num_heads, self.depth, 3])
        qkv = tf.transpose(qkv, [0, 4, 2, 1, 3])
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], attn.dtype)
        attn = attn / tf.math.sqrt(dk)
        attn = safe_softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, (B, N, self.d_model))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(Layer):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_channels, channels, out_channels=None, act_layer='gelu', drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        self.fc1 = Linear(in_channels, channels)
        self.act = Act(act_layer)
        self.fc2 = Linear(channels, out_channels)
        self.drop = Dropout(drop) if drop > 0 else Identity()

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, drop=0, attn_drop=0,
                 drop_path=0, act_layer='gelu'):
        super().__init__()

        self.ln1 = Norm(type='ln')
        self.mha = MultiHeadAttention(d_model, num_heads, attn_drop, drop)

        self.ln2 = Norm(type='ln')
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

    def call(self, x):
        x = x + self.drop_path(self.mha(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
