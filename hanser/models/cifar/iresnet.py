from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dropout

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Norm, Pool2d


class BasicBlock(Layer):
    def __init__(self, in_channels, out_channels, stride, drop_rate,
                 start_block=False, end_block=False, exclude_bn0=False):
        super().__init__()
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)

        if not start_block:
            self.act0 = Act()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn1 = Norm(out_channels)
        self.act1 = Act()
        self.dropout = Dropout(drop_rate) if drop_rate else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if start_block:
            self.bn2 = Norm(out_channels)

        if end_block:
            self.bn2 = Norm(out_channels)
            self.act2 = Norm(out_channels)

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
        x = self.dropout(x)
        x = self.conv2(x)
        if self.start_block:
            x = self.bn2(x)
        x = x + identity
        if self.end_block:
            x = self.bn2(x)
            x = self.act2(x)
        return x


class Bottleneck(Layer):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, erase_relu):
        super().__init__()
        channels = out_channels // self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='def')
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, norm='def')
        else:
            self.shortcut = Identity()
        self.act = Act() if not erase_relu else Identity()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNet(Model):

    def __init__(self, depth, block='basic', drop_rate=0.3, num_classes=10, stages=(16, 16, 32, 64)):
        super().__init__()
        self.stages = stages
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.conv = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            block, self.stages[0], self.stages[1], layers[0], stride=1,
            drop_rate=drop_rate)
        self.layer2 = self._make_layer(
            block, self.stages[1], self.stages[2], layers[1], stride=2,
            drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block, self.stages[2], self.stages[3], layers[2], stride=2,
            drop_rate=drop_rate)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3], num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride, drop_rate):
        layers = [block(in_channels, out_channels, stride=stride, start_block=True,
                        drop_rate=drop_rate)]
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                drop_rate=drop_rate))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
