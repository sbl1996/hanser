from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel, SplAtConv2d, DropPath

__all__ = [
    "PyramidNeSt"
]


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg'))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels)))
        super().__init__(layers)


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, groups, stride=1, radix=1, drop_path=0.2):
        super().__init__()
        out_channels = channels * self.expansion
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=1, norm='def', act='default'),
            *([Pool2d(3, 2)] if stride != 1 else []),
            SplAtConv2d(channels, channels, kernel_size=3, groups=groups, radix=radix)
            if radix != 0 else Conv2d(channels, channels, kernel_size=3, groups=groups, norm='def', act='default'),
            Conv2d(channels, out_channels, kernel_size=1, norm='def'),
            *([DropPath(drop_path)] if drop_path and stride == 1 else []),
        ]
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, out_channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class PyramidNeSt(Model):
    def __init__(self, start_channels, widening_fractor, depth, groups, radix, drop_path, num_classes=10):
        super().__init__()

        num_layers = [(depth - 2) // 9] * 3

        strides = [1, 2, 2]

        self.add_channel = widening_fractor / sum(num_layers)
        self.in_channels = start_channels
        self.channels = start_channels

        layers = [Conv2d(3, start_channels, kernel_size=3, norm='def')]

        for i, (n, s) in enumerate(zip(num_layers, strides)):
            layers.append(self._make_layer(n, groups, stride=s, radix=radix, drop_path=drop_path))

        layers.append(Sequential([
            Norm(self.in_channels),
            Act(),
        ]))

        self.features = Sequential(layers)
        assert (start_channels + widening_fractor) * Bottleneck.expansion == self.in_channels
        self.final_pool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, num_layers, groups, stride, radix, drop_path):
        self.channels = self.channels + self.add_channel
        layers = [
            Bottleneck(self.in_channels, round_channels(self.channels, groups * radix),
                       groups, stride=stride, radix=radix, drop_path=drop_path)]
        self.in_channels = round_channels(self.channels, groups * radix) * Bottleneck.expansion
        for i in range(1, num_layers):
            self.channels = self.channels + self.add_channel
            layers.append(Bottleneck(self.in_channels, round_channels(self.channels, groups * radix),
                                     groups, radix=radix, drop_path=drop_path))
            self.in_channels = round_channels(self.channels, groups * radix) * Bottleneck.expansion
        return Sequential(layers)

    def call(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = self.fc(x)
        return x