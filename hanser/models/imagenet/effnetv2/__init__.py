from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.modules import DropPath, Dropout
from hanser.models.attention import SELayer

from hanser.models.layers import Conv2d, Identity, GlobalAvgPool, Linear

v2_s_block = [
  # f,  e, k, se,   s, c,   l
    [1, 1, 3, None, 1, 24,  2],
    [1, 4, 3, None, 2, 48,  4],
    [1, 4, 3, None, 2, 64,  4],
    [0, 4, 3, 0.25, 2, 128, 6],
    [0, 6, 3, 0.25, 1, 160, 9],
    [0, 6, 3, 0.25, 2, 272, 15],
]


class MBConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, drop_connect):
        super().__init__()
        self._has_se = se_ratio is not None and 0 < se_ratio <= 1

        channels = in_channels * expand_ratio

        self.expand = Conv2d(in_channels, channels, 1,
                             norm='def', act='def') if expand_ratio != 1 else Identity()

        self.depthwise = Conv2d(channels, channels, kernel_size, stride, groups=channels,
                                norm='def', act='def')

        if self._has_se:
            self.se = SELayer(channels, se_channels=int(in_channels * se_ratio), min_se_channels=1)

        self.project = Conv2d(channels, out_channels, 1, norm='def')
        self._use_residual = in_channels == out_channels and stride == 1
        if self._use_residual:
            self.drop_connect = DropPath(drop_connect) if drop_connect else Identity()

    def call(self, x):
        identity = x

        x = self.expand(x)

        x = self.depthwise(x)

        if self._has_se:
            x = self.se(x)

        x = self.project(x)

        if self._use_residual:
            x = self.drop_connect(x)
            x = x + identity
        return x


class FusedMBConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, drop_connect):
        super().__init__()
        self._has_se = se_ratio is not None and 0 < se_ratio <= 1

        channels = in_channels * expand_ratio

        self.expand = Conv2d(in_channels, channels, kernel_size, stride,
                             norm='def', act='def') if expand_ratio != 1 else Identity()

        if self._has_se:
            self.se = SELayer(channels, se_channels=int(in_channels * se_ratio), min_se_channels=1)

        self.project = Conv2d(channels, out_channels, 1, norm='def')
        self._use_residual = in_channels == out_channels and stride == 1
        if self._use_residual:
            self.drop_connect = DropPath(drop_connect) if drop_connect else Identity()

    def call(self, x):
        identity = x

        x = self.expand(x)

        if self._has_se:
            x = self.se(x)

        x = self.project(x)

        if self._use_residual:
            x = self.drop_connect(x)
            x = x + identity
        return x


class EfficientNetV2(Model):

    def __init__(self, blocks_args, dropout, drop_connect=0.2, num_classes=1000):
        super().__init__()
        in_channels = blocks_args[0][-2]
        self.stem = Conv2d(3, in_channels, 3, stride=2, norm='def', act='def')

        blocks = []
        b = 0
        n_blocks = float(sum(args[-1] for args in blocks_args))
        for f, e, k, se, s, c, l in blocks_args:
            out_channels = c
            for j in range(l):
                stride = s if j == 0 else 1
                if f:
                    block = FusedMBConv(in_channels, out_channels, k, stride,
                                        e, se, drop_connect * b / n_blocks)
                else:
                    block = MBConv(in_channels, out_channels, k, stride,
                                   e, se, drop_connect * b / n_blocks)
                blocks.append(block)
                in_channels = out_channels
                b += 1

        self.blocks = Sequential(blocks)
        self.top = Conv2d(in_channels, 1280, 1, norm='def', act='def')
        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout)
        self.fc = Linear(1280, num_classes)

    def call(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.top(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def efficientnet_v2_s(dropout=0.2, **kwargs):
    return EfficientNetV2(v2_s_block, dropout=dropout, **kwargs)