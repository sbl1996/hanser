from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Identity, GlobalAvgPool, Linear, Pool2d


class SELayer(Layer):

    def __init__(self, in_channels, channels, **kwargs):
        super().__init__(**kwargs)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.fc = Sequential([
            Conv2d(in_channels, channels, 1, act='def'),
            Conv2d(channels, in_channels, 1, act='sigmoid'),
        ])

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups, se_channels):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                            norm='def', act='def')
        self.se = SELayer(out_channels, se_channels)
        self.conv3 = Sequential([
            Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            Norm(out_channels, gamma_init='zeros')
        ])
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
        x = self.se(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class RegNet(Model):

    def __init__(self, stem_channels, stages, layers, channels_per_group, se_reduction, num_classes=1000):
        super().__init__()
        block = Bottleneck

        self.stem = Conv2d(3, stem_channels, kernel_size=3, stride=2,
                           norm='def', act='def')
        self.in_channels = stem_channels
        gs = [c // channels_per_group for c in stages]

        self.stage1 = self._make_layer(
            block, stages[0], layers[0], stride=2, groups=gs[0], reduction=se_reduction)
        self.stage2 = self._make_layer(
            block, stages[1], layers[1], stride=2, groups=gs[1], reduction=se_reduction)
        self.stage3 = self._make_layer(
            block, stages[2], layers[2], stride=2, groups=gs[2], reduction=se_reduction)
        self.stage4 = self._make_layer(
            block, stages[3], layers[3], stride=2, groups=gs[3], reduction=se_reduction)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, groups, reduction):
        layers = [block(self.in_channels, channels, stride=stride,
                        groups=groups, se_channels=self.in_channels // reduction)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                groups=groups, se_channels=self.in_channels // reduction))
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
    return RegNet(32, (24, 56, 152, 368), (1, 1, 4, 7), 8, 4)

def regnety_400MF():
    return RegNet(32, (48, 104, 208, 440), (1, 3, 6, 6), 8, 4)

def regnety_600MF():
    return RegNet(32, (48, 112, 256, 608), (1, 3, 7, 4), 16, 4)

def regnety_800MF():
    return RegNet(32, (64, 128, 320, 768), (1, 3, 8, 2), 16, 4)

def regnety_1_6GF():
    return RegNet(32, (48, 120, 336, 888), (2, 6, 17, 2), 24, 4)

def regnety_3_2GF():
    return RegNet(32, (72, 216, 576, 1512), (2, 5, 13, 1), 24, 4)

def regnety_4_0GF():
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4)

def regnety_6_4GF():
    return RegNet(32, (144, 288, 576, 1296), (2, 7, 14, 2), 72, 4)

# def regnety_8_0GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4)
#
# def regnety_12GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4)
#
# def regnety_16GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4)
#
# def regnety_32GF():
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4)
