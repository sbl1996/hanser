import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel

__all__ = [
    "PyramidNet"
]


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride, name):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg', name="pool"))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels, name="pad")))
        super().__init__(layers, name=name)


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, name=None):
        super().__init__(name=name)
        branch1 = [
            Norm(in_channels, name="norm0"),
            Conv2d(in_channels, channels, kernel_size=3, stride=stride,
                   norm='default', act='default', name="conv1"),
            Conv2d(channels, channels, kernel_size=3,
                   norm='default', name="conv2"),
        ]
        self.branch1 = Sequential(branch1, name="branch1")
        self.branch2 = Shortcut(in_channels, channels, stride, name="branch2")

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, name=None):
        super().__init__(name=name)
        out_channels = channels * self.expansion
        branch1 = [
            Norm(in_channels, name="norm0"),
            Conv2d(in_channels, channels, kernel_size=1,
                   norm='default', act='default', name="conv1"),
            Conv2d(channels, channels, kernel_size=3, stride=stride,
                   norm='default', act='default', name="conv2"),
            Conv2d(channels, out_channels, kernel_size=1,
                   norm='default', name="conv3"),
        ]
        self.branch1 = Sequential(branch1, name="branch1")
        self.branch2 = Shortcut(in_channels, out_channels, stride, name="branch2")

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def rd(c):
    return int(round(c, 2))


class PyramidNet(Model):
    def __init__(self, start_channels, widening_fractor, depth, block='bottleneck', num_classes=10):
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

        add_channel = widening_fractor / sum(num_layers)
        in_channels = start_channels

        self.init_block = Conv2d(3, start_channels, kernel_size=3, norm='default', name="init_block")

        channels = start_channels
        k = 1
        units = []
        for i, (n, s) in enumerate(zip(num_layers, strides)):
            channels += add_channel
            units.append(block(in_channels, rd(channels), stride=s, name=f"unit{k}"))
            in_channels = rd(channels) * block.expansion
            k += 1

            for j in range(1, n):
                channels = channels + add_channel
                units.append(block(in_channels, rd(channels), name=f"unit{k}"))
                in_channels = rd(channels) * block.expansion
                k += 1

        self.units = units
        self.post_activ = Sequential([
            Norm(in_channels, name="norm"),
            Act(name="act"),
        ], name="post_activ")

        assert (start_channels + widening_fractor) * block.expansion == in_channels

        self.final_pool = GlobalAvgPool(name="final_pool")
        self.fc = Linear(in_channels, num_classes, name="fc")

    def call(self, x):
        x = self.init_block(x)
        for i, unit in enumerate(self.units):
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