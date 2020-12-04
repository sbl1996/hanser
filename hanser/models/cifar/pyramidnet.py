import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel

__all__ = [
    "PyramidNet"
]


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg'))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels)))
        super().__init__(layers)


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=3, stride=stride,
                   norm='def', act='def'),
            Conv2d(channels, channels, kernel_size=3, norm='def'),
        ]
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=1,
                   norm='def', act='def'),
            Conv2d(channels, channels, kernel_size=3, stride=stride,
                   norm='def', act='def'),
            Conv2d(channels, out_channels, kernel_size=1,
                   norm='def'),
        ]
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, out_channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def rd(c):
    return int(round(c, 2))


class PyramidNet(Model):
    def __init__(self, start_channels, alpha, depth, block='bottleneck', num_classes=10):
        super().__init__()

        if block == 'basic':
            num_layers = [(depth - 2) // 6] * 3
            block = BasicBlock
        elif block == 'bottleneck':
            num_layers = [(depth - 2) // 9] * 3
            block = Bottleneck
        else:
            raise ValueError("block must be `basic` or `bottleneck`, got %s" % block)

        self.num_layers = num_layers

        strides = [1, 2, 2]

        add_channel = alpha / sum(num_layers)
        in_channels = start_channels

        self.init_block = Conv2d(3, start_channels, kernel_size=3, norm='def')

        channels = start_channels
        k = 1
        units = []
        for i, (n, s) in enumerate(zip(num_layers, strides)):
            channels += add_channel
            units.append(block(in_channels, rd(channels), stride=s))
            in_channels = rd(channels) * block.expansion
            k += 1

            for j in range(1, n):
                channels = channels + add_channel
                units.append(block(in_channels, rd(channels)))
                in_channels = rd(channels) * block.expansion
                k += 1

        self.units = units
        self.post_activ = Sequential([
            Norm(in_channels),
            Act(),
        ])

        assert (start_channels + alpha) * block.expansion == in_channels

        self.final_pool = GlobalAvgPool()
        self.fc = Linear(in_channels, num_classes)

    def call(self, x):
        x = self.init_block(x)
        for unit in self.units:
            x = unit(x)
        x = self.post_activ(x)

        x = self.final_pool(x)
        x = self.fc(x)
        return x


def test_net():
    model = PyramidNet(16, 270, 164, 'bottleneck')
    model.build((None, 32, 32, 3))
    model.call(tf.keras.layers.Input((32, 32, 3)))
    model.summary()