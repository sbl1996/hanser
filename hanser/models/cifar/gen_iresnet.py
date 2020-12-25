from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Norm, Pool2d


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, out_channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 conv_cls=Conv2d):
        super().__init__()
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)
        if not start_block:
            self.act0 = Act()
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1)
        self.bn1 = Norm(channels)
        self.act1 = Act()
        self.conv2 = conv_cls(channels, channels, kernel_size=3, stride=stride,
                              norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

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

    def __init__(self, depth, num_classes=10, stages=(64, 64, 128, 256),
                 conv_channels=None, conv_cls=Conv2d):
        super().__init__()
        self.stages = stages
        self.conv_channels = conv_channels or stages
        self.conv_cls = conv_cls
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3
        cs = [stages[0]] + [c * block.expansion for c in stages[1:]]
        conv_cs = self.conv_channels

        self.conv = Conv2d(3, cs[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            block, cs[0], conv_cs[0], cs[1], layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, cs[1], conv_cs[1], cs[2], layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, cs[2], conv_cs[2], cs[3], layers[2], stride=2)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(cs[3], num_classes)

    def _make_layer(self, block, in_channels, conv_channels, out_channels, blocks, stride):
        layers = [block(in_channels, conv_channels, out_channels, stride=stride, start_block=True,
                        conv_cls=self.conv_cls)]
        for i in range(1, blocks):
            layers.append(block(out_channels, conv_channels, out_channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                conv_cls=self.conv_cls))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
