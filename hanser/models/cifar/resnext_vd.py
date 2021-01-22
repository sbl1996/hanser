import math
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Pool2d, Norm


class NaiveGroupConv2d(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, norm, act):
        super().__init__()
        self.groups = groups
        D_in = in_channels // groups
        D_out = out_channels // groups
        self.convs = [
            Conv2d(D_in, D_out, kernel_size=kernel_size, stride=stride)
            for _ in range(groups)
        ]
        self.norm = Norm(out_channels, norm) if norm is not None else None
        self.act = Act(act) if act is not None else None

    def call(self, x):
        xs = tf.split(x, self.groups, axis=-1)
        xs = [
            conv(x) for conv, x in zip(self.convs, xs)
        ]
        x = tf.concat(xs, axis=-1)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, cardinality, base_width, zero_init_residual=True):
        super().__init__()
        out_channels = channels * self.expansion

        D = math.floor(channels * (base_width / 64))
        C = cardinality

        self.conv1 = Conv2d(in_channels, D * C, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = NaiveGroupConv2d(
            D * C, D * C, kernel_size=3, stride=stride, groups=cardinality,
            norm='def', act='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1,
                            norm='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros' if zero_init_residual else 'ones')

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNeXt(Model):

    def __init__(self, depth, cardinality, base_width,
                 zero_init_residual=True, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3,
                           norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            cardinality=cardinality, base_width=base_width,
            zero_init_residual=zero_init_residual)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            cardinality=cardinality, base_width=base_width,
            zero_init_residual=zero_init_residual)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            cardinality=cardinality, base_width=base_width,
            zero_init_residual=zero_init_residual)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
