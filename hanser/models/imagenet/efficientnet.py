import math
from functools import partial
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from hanser.models.modules import SELayer, DropPath
from tensorflow.keras.layers import Layer, Dropout
from hanser.models.layers import Conv2d, Identity, GlobalAvgPool, Linear

params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}

blocks_args = [
    # r, k, s, e,  i,   o,   se
    [1, 3, 1, 1, 32, 16, 0.25],
    [2, 3, 2, 6, 16, 24, 0.25],
    [2, 5, 2, 6, 24, 40, 0.25],
    [3, 3, 2, 6, 40, 80, 0.25],
    [3, 5, 1, 6, 80, 112, 0.25],
    [4, 5, 2, 6, 112, 192, 0.25],
    [1, 3, 1, 6, 192, 320, 0.25],

]


def round_channels(channels, width_coefficient, depth_divisor=8, skip=False):
    multiplier = width_coefficient
    divisor = depth_divisor
    if skip or not multiplier:
        return channels

    channels *= multiplier
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def round_repeats(repeats, depth_coefficient, skip=False):
    multiplier = depth_coefficient
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MBConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, drop_connect):
        super().__init__()
        self._has_se = se_ratio is not None and 0 < se_ratio <= 1

        channels = in_channels * expand_ratio

        self.expand = Conv2d(in_channels, channels, 1,
                             norm='def', act='def') if expand_ratio != 1 else Identity()

        self.depthwise = Conv2d(channels, channels, kernel_size, stride, groups=channels, padding='SAME',
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


class EfficientNet(Model):

    def __init__(self, width_coefficient, depth_coefficient, dropout, drop_connect=0.3, depth_divisor=8,
                 num_classes=1000):
        super().__init__()
        round_channels_ = partial(
            round_channels, width_coefficient=width_coefficient, depth_divisor=depth_divisor)
        round_repeats_ = partial(round_repeats, depth_coefficient=depth_coefficient)
        self.stem = Conv2d(3, round_channels_(32), 3, stride=2, padding='SAME',
                           norm='def', act='def')

        blocks = []
        b = 0
        n_blocks = float(sum(round_repeats_(args[0]) for args in blocks_args))
        for r, k, s, e, i, o, se in blocks_args:
            in_channels = round_channels_(i)
            out_channels = round_channels_(o)

            for j in range(round_repeats_(r)):
                stride = s if j == 0 else 1
                blocks.append(
                    MBConv(in_channels, out_channels, k, stride,
                           e, se, drop_connect * b / n_blocks)
                )
                in_channels = out_channels
                b += 1

        self.blocks = Sequential(blocks)
        self.top = Conv2d(out_channels, round_channels_(1280), 1,
                   norm='def', act='def')
        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout)
        self.fc = Linear(round_channels_(1280), num_classes)

    def call(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.top(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def efficientnet_b0(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b0']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b1(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b1']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b2(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b2']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b3(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b3']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b4(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b4']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b5(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b5']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b6(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b6']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b7(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b7']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)

def efficientnet_b8(**kwargs):
    width_coefficient, depth_coefficient, resolution, dropout = params_dict['efficientnet-b8']
    return EfficientNet(width_coefficient, depth_coefficient, dropout, **kwargs)
