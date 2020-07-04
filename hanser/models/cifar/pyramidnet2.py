import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from hanser.models.layers import Pool2d, Conv2d, BN, Act, GlobalAvgPool, Linear, Sequential
from hanser.models.modules import PadChannel

__all__ = [
    "PyramidNet"
]


class Shortcut(Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = []
        if stride == 2:
            layers.append(("pool", Pool2d(2, 2)))
        if in_channels != out_channels:
            layers.append(("pad", PadChannel(out_channels - in_channels)))
        self.shortcut = Sequential(*layers)

    def call(self, x):
        return self.shortcut(x)


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, groups, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        branch1 = [
            ("bn1", BN(in_channels, name="bn1")),
            ("conv1", Conv2d(in_channels, channels, kernel_size=1, bias=False)),
            ("bn2", BN(channels)),
            ("act2", Act()),
            ("conv2", Conv2d(channels, channels, kernel_size=3, stride=stride, bias=False, groups=groups)),
            ("bn3", BN(channels)),
            ("act3", Act()),
            ("conv3", Conv2d(channels, out_channels, kernel_size=1, bias=False)),
            ("bn4", BN(out_channels)),
        ]
        self.branch1 = Sequential(*branch1)
        self.branch2 = Shortcut(in_channels, out_channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class PyramidNet(Model):
    def __init__(self, start_channels, widening_fractor, depth, groups, num_classes):
        super().__init__()

        num_layers = [(depth - 2) // 9] * 3

        strides = [1, 2, 2]

        self.add_channel = widening_fractor / sum(num_layers)
        self.in_channels = start_channels
        self.channels = start_channels

        layers = [
            ("conv", Conv2d(3, start_channels, kernel_size=3)),
        ]

        for i, (n, s) in enumerate(zip(num_layers, strides)):
            layers.append(
                ("stage%d" % (i + 1), self._make_layer(n, groups, stride=s))
            )

        layers.append(
            ("post_activ", Sequential(
                ("bn", BN(self.in_channels)),
                ("act", Act()),
            ))
        )

        self.features = Sequential(*layers)
        assert (start_channels + widening_fractor) * Bottleneck.expansion == self.in_channels
        self.final_pool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, num_layers, groups, stride):
        self.channels = self.channels + self.add_channel
        layers = [("unit1", Bottleneck(self.in_channels, round_channels(self.channels, groups), groups, stride=stride))]
        self.in_channels = round_channels(self.channels, groups) * Bottleneck.expansion
        for i in range(1, num_layers):
            self.channels = self.channels + self.add_channel
            layers.append(
                ("unit%d" % (i + 1), Bottleneck(self.in_channels, round_channels(self.channels, groups), groups)))
            self.in_channels = round_channels(self.channels, groups) * Bottleneck.expansion
        return Sequential(*layers)

    def call(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = self.fc(x)
        return x

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        self.call(inputs)
