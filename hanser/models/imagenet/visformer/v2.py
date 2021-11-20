import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import TruncatedNormal, Zeros

from einops import rearrange

from hanser.models.layers import Conv2d, Dropout, Identity, Norm, GlobalAvgPool, Linear
from hanser.models.modules import DropPath

__all__=[
    'visformer_small', 'visformer_tiny',
]


class Mlp(Layer):

    def __init__(self, in_channels, channels=None, out_channels=None,
                 dropout=0., group=8, spatial_conv=False):
        super().__init__()
        out_channels = out_channels or in_channels
        channels = channels or in_channels
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            channels = in_channels * 2
        self.channels = channels
        self.group = group
        self.drop = Dropout(dropout)
        self.conv1 = Conv2d(in_channels, channels, 1, bias=False, act='def')
        if self.spatial_conv:
            self.conv2 = Conv2d(channels, channels, 3, groups=self.group, bias=False, act='def')
        self.conv3 = Conv2d(channels, out_channels, 1, bias=False)

    def call(self, x):
        x = self.conv1(x)
        x = self.drop(x)

        if self.spatial_conv:
            x = self.conv2(x)

        x = self.conv3(x)
        x = self.drop(x)
        return x


class Attention(Layer):

    def __init__(self, dim, num_heads=8, head_dim_ratio=1., attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        self.scale = head_dim ** -0.25

        self.qkv = Conv2d(dim, head_dim * num_heads * 3, 1, bias=False)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Conv2d(self.head_dim * self.num_heads, dim, 1, bias=False)
        self.proj_drop = Dropout(proj_drop)

    def call(self, x):
        H, W = x.shape[1:3]
        x = self.qkv(x)

        qkv = rearrange(x, 'b h w (x y z) -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q * self.scale, k * self.scale, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)

        x = rearrange(x, 'b y (h w) z -> b h w (y z)', h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(Layer):
    def __init__(self, dim, num_heads, head_dim_ratio=1., mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0., group=8, attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if not attn_disabled:
            self.norm1 = Norm(dim)
            self.attn = Attention(dim, num_heads, head_dim_ratio, attn_drop, drop)

        self.norm2 = Norm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, dropout=drop, group=group, spatial_conv=spatial_conv)

    def call(self, x):
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(Layer):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, norm='def')

    def call(self, x):
        x = self.proj(x)
        return x


class PositionEmbedding(Layer):

    def __init__(self, shape, dropout=0.):
        super().__init__()
        self.embedding = self.add_weight(
            name="embedding", shape=shape, dtype=self.dtype,
            initializer=TruncatedNormal(stddev=0.02), trainable=True,
        )
        self.dropout = Dropout(dropout) if dropout else None

    def call(self, x):
        x = x + self.embedding
        if self.dropout:
            x = self.dropout(x)
        return x


class Visformer(Model):

    def __init__(self, img_size=224, init_channels=32, num_classes=1000, embed_dim=384, depth=(7, 4, 4),
                 num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path=0.,
                 attn_stage=(False, True, True), spatial_conv=(True, False, False), group=8):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.stage_num1, self.stage_num2, self.stage_num3 = depth
        depth = sum(depth)
        dpr = [x.numpy() for x in tf.linspace(0., drop_path, depth)]

        self.stem = Conv2d(3, init_channels, 7, stride=2, norm='def', act='relu')
        img_size //= 2

        # stage 1
        self.patch_embed1 = PatchEmbed(4, self.init_channels, embed_dim//2)
        img_size //= 4
        self.pos_embed1 = PositionEmbedding((img_size, img_size, embed_dim//2), drop_rate)
        self.stage1 = Sequential([
            Block(
                dim=embed_dim//2, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=not attn_stage[0], spatial_conv=spatial_conv[0]
            )
            for i in range(self.stage_num1)
        ])

        #stage2
        self.patch_embed2 = PatchEmbed(2, embed_dim//2, embed_dim)
        img_size //= 2
        self.pos_embed2 = PositionEmbedding((img_size, img_size, embed_dim), drop_rate)
        self.stage2 = Sequential([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=not attn_stage[1], spatial_conv=spatial_conv[1]
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        self.patch_embed3 = PatchEmbed(2, embed_dim, embed_dim*2)
        img_size //= 2
        self.pos_embed3 = PositionEmbedding((img_size, img_size, embed_dim*2), drop_rate)
        self.stage3 = Sequential([
            Block(
                dim=embed_dim*2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=not attn_stage[2], spatial_conv=spatial_conv[2]
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        self.norm = Norm(embed_dim*2)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(embed_dim*2, num_classes, kernel_init=TruncatedNormal(stddev=0.02), bias_init=Zeros())

    def call(self, x):
        x = self.stem(x)

        # stage 1
        x = self.patch_embed1(x)
        x = self.pos_embed1(x)
        x = self.stage1(x)

        # stage 2
        x = self.patch_embed2(x)
        x = self.pos_embed2(x)
        x = self.stage2(x)

        # stage3
        x = self.patch_embed3(x)
        x = self.pos_embed3(x)
        x = self.stage3(x)

        x = self.norm(x)

        x = self.avgpool(x)

        x = self.fc(x)
        return x


def visformer_tiny(drop_path=0.03, **kwargs):
    model = Visformer(
        img_size=224, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4.,
        group=8, drop_path=drop_path, **kwargs)
    return model


def visformer_small(drop_path=0.1, **kwargs):
    model = Visformer(
        img_size=224, init_channels=32, embed_dim=384, depth=[7,4,4], num_heads=6, mlp_ratio=4.,
        group=8, drop_path=drop_path, **kwargs)
    return model