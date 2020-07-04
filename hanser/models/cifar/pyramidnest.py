from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Pool2d, Conv2d, BN, Act, GlobalAvgPool, Linear, Sequential
from hanser.models.modules import PadChannel, SplAtConv2d

__all__ = [
    "PyramidNeSt"
]


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride, name):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, name=name + "/pool"))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels, name=name + "/pad")))
        super().__init__(layers, name=name)


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, groups, stride=1, radix=2, name=None):
        super().__init__(name=name)
        out_channels = channels * self.expansion
        branch1 = [
            BN(in_channels, name=name + "bn0"),
            Conv2d(in_channels, channels, kernel_size=1, bn=True, act='default', name=name + "conv1"),
            *([Pool2d(3, 2, name=name + "pool")] if stride != 1 else []),
            SplAtConv2d(channels, channels, kernel_size=3, groups=groups, radix=radix, name=name + "conv2")
            if radix != 0 else Conv2d(channels, channels, kernel_size=3, groups=groups, bn=True, act='default', name=name + "conv2"),
            Conv2d(channels, out_channels, kernel_size=1, bn=True, name=name + "conv3"),
        ]
        self.branch1 = Sequential(branch1, name=name + "/branch1")
        self.branch2 = Shortcut(in_channels, out_channels, stride, name=name + "/branch2")

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class PyramidNeSt(Model):
    def __init__(self, start_channels, widening_fractor, depth, groups, radix, num_classes=10):
        super().__init__()

        num_layers = [(depth - 2) // 9] * 3

        strides = [1, 2, 2]

        self.add_channel = widening_fractor / sum(num_layers)
        self.in_channels = start_channels
        self.channels = start_channels

        layers = [Conv2d(3, start_channels, kernel_size=3, bn=True, name="init_block")]

        for i, (n, s) in enumerate(zip(num_layers, strides)):
            layers.append(self._make_layer(n, groups, stride=s, radix=radix, name=f"stage{i + 1}"))

        layers.append(Sequential([
            BN(self.in_channels, name="post_activ/bn"),
            Act(name="post_activ/act"),
        ], name="post_activ"))

        self.features = Sequential(layers, name="features")
        assert (start_channels + widening_fractor) * Bottleneck.expansion == self.in_channels
        self.final_pool = GlobalAvgPool(name="final_pool")
        self.fc = Linear(self.in_channels, num_classes, name="fc")

    def _make_layer(self, num_layers, groups, stride, radix, name):
        self.channels = self.channels + self.add_channel
        layers = [
            Bottleneck(self.in_channels, round_channels(self.channels, groups * radix),
                       groups, stride=stride, radix=radix, name=name + "/unit1")]
        self.in_channels = round_channels(self.channels, groups * radix) * Bottleneck.expansion
        for i in range(1, num_layers):
            self.channels = self.channels + self.add_channel
            layers.append(Bottleneck(self.in_channels, round_channels(self.channels, groups * radix),
                                     groups, radix=radix, name=f"{name}/unit{i + 1}"))
            self.in_channels = round_channels(self.channels, groups * radix) * Bottleneck.expansion
        return Sequential(layers, name=name)

    def call(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = self.fc(x)
        return x