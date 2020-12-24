import math
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Norm, Pool2d
from hanser.models.cifar.res2net.layers import Res2Conv


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, base_width, scale,
                 start_block=False, end_block=False, exclude_bn0=False):
        super().__init__()
        out_channels = channels * self.expansion
        width = math.floor(out_channels // self.expansion * (base_width / 64)) * scale
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)
        if not start_block:
            self.act0 = Act()
        self.conv1 = Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = Norm(width)
        self.act1 = Act()
        self.conv2 = Res2Conv(width, kernel_size=3, stride=stride, scale=scale,
                              norm='def', act='def', start_block=start_block)
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if end_block:
            self.bn3 = Norm(out_channels)
            self.act3 = Norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()
        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x):
        identity = self.shortcut(x)
        if self.start_block:
            x = self.conv1(x)
        else:
            if not self.exclude_bn0:
                x = self.bn0(x)
            x = self.act0(x)
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.start_block:
            x = self.bn3(x)
        x = x + identity
        if self.end_block:
            x = self.bn3(x)
            x = self.act3(x)
        return x


class ResNet(Model):

    def __init__(self, depth, base_width=26, scale=4, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3

        self.conv = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            block, self.stages[0], self.stages[1], layers[0], stride=1,
            base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(
            block, self.stages[1], self.stages[2], layers[1], stride=2,
            base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(
            block, self.stages[2], self.stages[3], layers[2], stride=2,
            base_width=base_width, scale=scale)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3], num_classes)

    def _make_layer(self, block, in_channels, channels, blocks, stride, base_width, scale):
        layers = [block(in_channels, channels, stride=stride, start_block=True,
                        base_width=base_width, scale=scale)]
        out_channels = channels * 4
        for i in range(1, blocks):
            layers.append(block(out_channels, channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                base_width=base_width, scale=scale))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
