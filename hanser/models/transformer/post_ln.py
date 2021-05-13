from tensorflow.keras.layers import Layer

from hanser.models.layers import Norm
from hanser.models.transformer.modules import MultiHeadAttention, FFN


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate, attn_drop_rate, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.activation = activation

        self.mha = MultiHeadAttention(d_model, num_heads, attn_drop_rate)
        self.ln1 = Norm(type='ln')

        self.ffn = FFN(d_model, dff, drop_rate, activation)
        self.ln2 = Norm(type='ln')

    def call(self, x):
        identity = x
        x, _ = self.mha(x, x, x)
        x = x + identity
        x = self.ln1(x)

        identity = x
        x = self.ffn(x)
        x = x + identity
        x = self.ln2(x)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "activation": self.activation,
        }
