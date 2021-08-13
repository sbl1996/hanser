import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal

from hanser.models.layers import Norm, Conv2d, Linear, Identity, Dropout

from hanser.models.transformer.pre_ln import TransformerEncoderLayer


def resize_pos_embed(pos_embed, ori_shape, shape):
    seq_len, embed_dim = pos_embed.shape
    pos_embed_token = pos_embed[:1, :]
    pos_embed_grid = pos_embed[1:, :]

    pos_embed_grid = tf.reshape(pos_embed_grid, (*ori_shape, embed_dim))
    pos_embed_grid = tf.image.resize(pos_embed_grid, shape, method='bilinear')
    pos_embed_grid = tf.reshape(pos_embed_grid, (shape[0] * shape[1], embed_dim))

    pos_embed = tf.concat([pos_embed_token, pos_embed_grid], axis=0)
    return pos_embed


class VisionTransformer(Model):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 num_layers=12, d_model=192, num_heads=3, dff=768,
                 drop_rate=0, attn_drop_rate=0, drop_path=0.1, activation='gelu', with_head=True):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.num_classes = num_classes
        self.with_head = with_head

        self.f_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        num_patches = self.f_shape[0] * self.f_shape[1]
        self.patch_embed = Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, padding=0)

        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(num_patches + 1, d_model), dtype=tf.float32,
            initializer=TruncatedNormal(stddev=0.02), trainable=True,
        )
        self.pos_dropout = Dropout(drop_rate) if drop_rate else Identity()

        self.cls_token = self.add_weight(
            name="cls_token", shape=(d_model,), dtype=tf.float32,
            initializer=TruncatedNormal(stddev=0.02), trainable=True,
        )

        drop_path_rates = np.linspace(0, drop_path, num_layers)
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, drop_rate, attn_drop_rate,
                                    activation, float(drop_path_rates[i]))
            for i in range(num_layers)
        ]

        self.norm = Norm(type='ln')

        if with_head:
            self.cls_head = Linear(d_model, num_classes,
                                   kernel_init=TruncatedNormal(stddev=0.02))

    def forward_features(self, x):
        B = tf.shape(x)[0]

        x = self.patch_embed(x)
        f_shape, d_model = tuple(x.shape[1:3]), x.shape[3]
        num_patches = f_shape[0] * f_shape[1]
        x = tf.reshape(x, (B, num_patches, d_model))

        cls_token = tf.tile(self.cls_token[None, None, :], [B, 1, 1])
        cls_token = tf.cast(cls_token, x.dtype)
        x = tf.concat([cls_token, x], axis=1)

        if f_shape != self.f_shape:
            pos_embed = resize_pos_embed(self.pos_embed, self.f_shape, f_shape)
        else:
            pos_embed = self.pos_embed
        pos_embed = tf.cast(pos_embed, x.dtype)

        x = x + pos_embed
        x = self.pos_dropout(x)

        for l in self.enc_layers:
            x = l(x)
        x = self.norm(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.with_head:
            x = self.cls_head(x[:, 0, :])
        return x
