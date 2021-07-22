import math
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.attention import SELayer
from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear


class Bottleneck(Layer):

    def __init__(self, in_channels, channels, stride, cardinality, base_width, reduction):
        super().__init__()
        out_channels = channels * 4

        D = math.floor(channels * (base_width / 64))
        C = cardinality

        self.conv1 = Conv2d(in_channels, D * C, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(D * C, D * C, kernel_size=3, stride=stride, groups=cardinality,
                            norm='def', act='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1,
                            norm='def')
        self.se = SELayer(out_channels, reduction=reduction)
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='def') if in_channels != out_channels else Identity()
        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNeXt(Model):
    stages = [64, 64, 128, 256]

    def __init__(self, depth, cardinality, base_width, reduction=16, num_classes=10):
        super().__init__()
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            self.stages[0], self.stages[1], layers[0], 1, cardinality, base_width, reduction)
        self.layer2 = self._make_layer(
            self.stages[1], self.stages[2], layers[1], 2, cardinality, base_width, reduction)
        self.layer3 = self._make_layer(
            self.stages[2], self.stages[3], layers[2], 2, cardinality, base_width, reduction)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3], num_classes)

    def _make_layer(self, in_channels, channels, blocks, stride, cardinality, base_width, reduction):
        layers = [Bottleneck(in_channels, channels, stride, cardinality, base_width, reduction)]
        out_channels = channels * 4
        for i in range(1, blocks):
            layers.append(Bottleneck(out_channels, channels, 1, cardinality, base_width, reduction))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


