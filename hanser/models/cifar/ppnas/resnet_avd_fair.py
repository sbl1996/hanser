import math

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Linear, Pool2d, Identity, GlobalAvgPool

from hanser.models.cifar.res2net.layers import Res2Conv
from hanser.models.cifar.ppnas.operations import OPS

def parse_genotype(genotype):
    connections = []
    op_types = []
    for g in genotype:
        connections.append(g[:-1])
        op_types.append(g[-1])
    return connections, op_types

class PPConv(Layer):

    def __init__(self, channels, splits, genotype, dilation=1):
        super().__init__()
        self.splits = splits
        C = channels // splits

        assert len(genotype) == splits
        connections, op_types = parse_genotype(genotype)

        self.ops = []
        self.connections = connections
        for i in range(self.splits):
            op_type = op_types[i]
            if op_type == 'nor_conv_3x3':
                op = OPS[op_type](C, 1, dilation)
            else:
                op = OPS[op_type](C, 1)
            self.ops.append(op)

    def call(self, x):
        states = list(tf.split(x, self.splits, axis=-1))
        for i in range(self.splits):
            x = sum(states[j-1] for j in self.connections[i]) / tf.convert_to_tensor(len(self.connections[i]), x.dtype)
            x = self.ops[i](x)
            states.append(x)
        return tf.concat(states[-self.splits:], axis=-1)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, base_width, splits, zero_init_residual, genotype, dilation=1):
        super().__init__()
        self.stride = stride

        out_channels = channels * self.expansion
        width = math.floor(out_channels // self.expansion * (base_width / 64)) * splits
        self.conv1 = Conv2d(in_channels, width, kernel_size=1,
                            norm='def', act='def')
        # start of stage
        if stride != 1 or in_channels != out_channels:
            layers = []
            if stride != 1:
                layers.append(Pool2d(3, stride=2, type='avg'))
            layers.append(
                Res2Conv(width, width, kernel_size=3, stride=1, dilation=dilation, scale=splits,
                         norm='def', act='def', start_block=True))
            self.conv2 = Sequential(layers)
        else:
            self.conv2 = PPConv(width, dilation=dilation, splits=splits, genotype=genotype)
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)
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


class ResNet(Model):

    def __init__(self, genotype, depth=110, base_width=26, splits=4, zero_init_residual=True,
                 num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3
        genotype = genotype.normal

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[0])
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[1])
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[2])

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, **kwargs))
        return Sequential(layers)

    def call(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x