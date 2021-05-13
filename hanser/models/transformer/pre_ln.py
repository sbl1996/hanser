import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Dropout, Embedding
from tensorflow.keras.initializers import Constant

from hanser.models.layers import Norm, Identity
from hanser.models.modules import DropPath
from hanser.models.transformer.modules import MultiHeadAttention, positional_encoding, FFN


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0, attn_drop_rate=0,
                 activation='gelu', drop_path=0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.drop_rate = drop_rate
        self.activation = activation
        self.drop_path = drop_path

        self.ln1 = Norm(type='ln')
        self.mha = MultiHeadAttention(d_model, num_heads, attn_drop_rate)

        self.ffn = FFN(d_model, dff, drop_rate, activation)
        self.ln2 = Norm(type='ln')

        self.drop_path = DropPath(drop_path) if drop_rate else Identity()

    def call(self, x):
        identity = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x)
        x = self.drop_path(x)
        x = x + identity

        identity = x
        x = self.ffn(self.ln2(x))
        x = self.drop_path(x)
        x = x + identity
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "drop_rate": self.drop_rate,
            "drop_path": self.drop_path,
            "activation": self.activation,
        }


class TransformerEncoder(Layer):
    def __init__(self, n_tokens, seq_len, num_layers=4, d_model=256, num_heads=8, dff=1024, drop_rate=0.1, activation='gelu',
                 with_head=False, **kwargs):
        super().__init__(**kwargs)

        self.n_tokens = n_tokens
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.drop_rate = drop_rate
        self.activation = activation
        self.with_head = with_head

        self.embedding = Embedding(n_tokens, d_model)
        self.pos_embedding = self.add_weight(
            name="pos_embedding", shape=(seq_len, self.d_model), dtype=self.dtype,
            initializer=Constant(positional_encoding(seq_len, d_model)), trainable=True,
        )
        self.dropout = Dropout(drop_rate) if drop_rate else None

        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, drop_rate, activation)
            for i in range(num_layers)
        ]

        self.ln = Norm(type='ln')

        if with_head:
            self.cls_head = Dense(n_tokens)

    def forward_features(self, x):
        x = self.embedding(x)
        x = tf.cast(x, self.pos_embedding.dtype)
        x += self.pos_embedding[:x.shape[1]]
        if self.dropout:
            x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        x = self.ln(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.with_head:
            x = self.cls_head(x)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "n_tokens": self.n_tokens,
            "seq_len": self.seq_len,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            "dff": self.dff,
            "drop_rate": self.drop_rate,
            "activation": self.activation,
            "with_head": self.with_head,
        }
