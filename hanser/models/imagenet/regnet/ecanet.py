from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Identity, GlobalAvgPool, Linear
from hanser.models.imagenet.stem import SimpleStem
from hanser.models.attention import ECALayer


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                            norm='def', act='def')
        self.eca = ECALayer(kernel_size=3)
        self.conv3 = Sequential([
            Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            Norm(out_channels, gamma_init='zeros')
        ])
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels, out_channels, stride=stride, kernel_size=1, norm='def')
        else:
            self.shortcut = Identity()
        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.eca(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class RegNet(Model):

    def __init__(self, stem_channels, stages, layers, channels_per_group, num_classes=1000, block=Bottleneck):
        super().__init__()

        self.stem = SimpleStem(stem_channels)
        self.in_channels = stem_channels
        gs = [c // channels_per_group for c in stages]

        self.stage1 = self._make_layer(
            block, stages[0], layers[0], stride=2, groups=gs[0])
        self.stage2 = self._make_layer(
            block, stages[1], layers[1], stride=2, groups=gs[1])
        self.stage3 = self._make_layer(
            block, stages[2], layers[2], stride=2, groups=gs[2])
        self.stage4 = self._make_layer(
            block, stages[3], layers[3], stride=2, groups=gs[3])

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, groups):
        layers = [block(self.in_channels, channels, stride=stride, groups=groups)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, groups=groups))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def regnety_200MF():
    return RegNet(32, (24, 56, 152, 368), (1, 1, 4, 7), 8, block=Bottleneck)

def regnety_400MF():
    return RegNet(32, (48, 104, 208, 440), (1, 3, 6, 6), 8, block=Bottleneck)

def regnety_600MF():
    return RegNet(32, (48, 112, 256, 608), (1, 3, 7, 4), 16, block=Bottleneck)

def regnety_800MF():
    return RegNet(32, (64, 128, 320, 768), (1, 3, 8, 2), 16, block=Bottleneck)

def regnety_1_6GF():
    return RegNet(32, (48, 120, 336, 888), (2, 6, 17, 2), 24, block=Bottleneck)

def regnety_3_2GF():
    return RegNet(32, (72, 216, 576, 1512), (2, 5, 13, 1), 24, block=Bottleneck)

def regnety_4_0GF():
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, block=Bottleneck)

def regnety_6_4GF():
    return RegNet(32, (144, 288, 576, 1296), (2, 7, 14, 2), 72, block=Bottleneck)

# def regnety_8_0GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, block=Bottleneck)
#
# def regnety_12GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, block=Bottleneck)
#
# def regnety_16GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, block=Bottleneck)
#
# def regnety_32GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, block=Bottleneck)
