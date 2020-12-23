from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear


class BasicBlock(Layer):
    def __init__(self, in_channels, out_channels, stride, erase_relu):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='def')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='def') if stride != 1 else Identity()
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

    def __init__(self, in_channels, out_channels, stride, erase_relu):
        super().__init__()
        channels = out_channels // self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='def')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='def') if stride != 1 else Identity()
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
    stages = [16, 16, 32, 64]

    def __init__(self, depth, block='basic', erase_relu=False, num_classes=10):
        super().__init__()
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.conv = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            block, self.stages[0], self.stages[1], layers[0], stride=1,
            erase_relu=erase_relu)
        self.layer2 = self._make_layer(
            block, self.stages[1], self.stages[2], layers[1], stride=2,
            erase_relu=erase_relu)
        self.layer3 = self._make_layer(
            block, self.stages[2], self.stages[3], layers[2], stride=2,
            erase_relu=erase_relu)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3], num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, erase_relu=False):
        layers = [block(in_channels, out_channels, stride=stride, erase_relu=erase_relu)]
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, erase_relu=erase_relu))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


