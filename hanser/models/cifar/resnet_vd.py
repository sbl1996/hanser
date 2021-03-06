from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Pool2d, Norm


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride, erase_relu=False, zero_init_residual=False):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='def')

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act() if not erase_relu else Identity()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, erase_relu=False, zero_init_residual=True, avd=False):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        if avd and stride != 1:
            self.conv2 = Sequential([
                Pool2d(3, stride, type='avg'),
                Conv2d(channels, channels, kernel_size=3, stride=1,
                       norm='def', act='def'),
            ])
        else:
            self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                                norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)
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

        self.act = Act() if not erase_relu else Identity()

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

    def __init__(self, depth, block='basic', erase_relu=False, avd=False, num_classes=10, stages=(16, 16, 32, 64)):
        super().__init__()
        self.stages = stages
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            erase_relu=erase_relu, avd=avd)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            erase_relu=erase_relu, avd=avd)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            erase_relu=erase_relu, avd=avd)

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


