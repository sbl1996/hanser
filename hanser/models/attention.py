import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import GlobalAvgPool, Conv2d


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class SELayer(Layer):

    def __init__(self, in_channels, reduction=None, groups=1, se_channels=None,
                 min_se_channels=32, act='def', mode=0, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.pool = GlobalAvgPool(keep_dim=True)
        if mode == 0:
            # σ(f_{W1, W2}(y))
            channels = se_channels or min(max(in_channels // reduction, min_se_channels), in_channels)
            if groups != 1:
                channels = round_channels(channels, groups)
            self.fc = Sequential([
                Conv2d(in_channels, channels, kernel_size=1, bias=False, act=act),
                Conv2d(channels, in_channels, 1, groups=groups, act='sigmoid'),
            ])
        elif mode == 1:
            # σ(w ⊙ y)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1,
                             groups=in_channels, bias=False, act='sigmoid')
        elif mode == 2:
            # σ(Wy)
            assert groups == 1
            self.fc = Conv2d(in_channels, in_channels, 1, bias=False, act='sigmoid')
        else:
            raise ValueError("Not supported mode: {}" % mode)

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        if self.scale is not None:
            s = s * tf.constant(self.scale, s.dtype)
        return x * s


# class CBAMChannelAttention(Layer):
#     def __init__(self, in_channels, reduction=8):
#         super().__init__()
#         channels = in_channels // reduction
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, channels),
#             nn.ReLU(True),
#             nn.Linear(channels, in_channels),
#         )
#
#     def forward(self, x):
#         b, c = x.size()[:2]
#         aa = F.adaptive_avg_pool2d(x, 1).view(b, c)
#         aa = self.mlp(aa)
#         am = F.adaptive_max_pool2d(x, 1).view(b, c)
#         am = self.mlp(am)
#         a = torch.sigmoid(aa + am).view(b, c, 1, 1)
#         return x * a
#
#
# class CBAMSpatialAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = Conv2d(2, 1, kernel_size=7, norm='bn')
#
#     def forward(self, x):
#         aa = x.mean(dim=1, keepdim=True)
#         am = x.max(dim=1, keepdim=True)[0]
#         a = torch.cat([aa, am], dim=1)
#         a = torch.sigmoid(self.conv(a))
#         return x * a
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super().__init__()
#         self.channel = CBAMChannelAttention(in_channels, reduction)
#         self.spatial = CBAMSpatialAttention()
#
#     def forward(self, x):
#         x = self.channel(x)
#         x = self.spatial(x)
#         return x
