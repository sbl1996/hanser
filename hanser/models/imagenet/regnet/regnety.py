from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear, Pool2d
from hanser.models.modules import Dropout, DropPath
from hanser.models.attention import SELayer
from hanser.models.imagenet.stem import SimpleStem


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups,
                 se_reduction=4, se_last=False, se_mode=0,
                 avg_shortcut=False, drop_path=0):
        super().__init__()
        self.se_last = se_last

        self.conv1 = Conv2d(in_channels, out_channels, 1, norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, 3, stride=stride, groups=groups,
                            norm='def', act='def')
        self.conv3 = Conv2d(out_channels, out_channels, 1, norm='def')
        self.se = SELayer(out_channels, reduction=se_reduction, mode=se_mode)
        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if stride != 1 or in_channels != out_channels:
            if avg_shortcut and stride != 1:
                self.shortcut = Sequential([
                    Pool2d(2, 2, type='avg'),
                    Conv2d(in_channels, out_channels, kernel_size=1, norm='def'),
                ])
            else:
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=stride, norm='def')
        else:
            self.shortcut = Identity()
        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if not self.se_last:
            x = self.se(x)
        x = self.conv3(x)
        if self.se_last:
            x = self.se(x)
        x = self.drop_path(x)
        x = x + identity
        x = self.act(x)
        return x


class RegNet(Model):

    def __init__(self, stem_channels, channels, layers, channels_per_group, se_reduction=4,
                 se_last=False, se_mode=0, avg_shortcut=False, dropout=0, num_classes=1000):
        super().__init__()
        block = Bottleneck

        self.stem = SimpleStem(stem_channels)
        self.in_channels = stem_channels
        gs = [c // channels_per_group for c in channels]

        for i in range(4):
            layer = self._make_layer(
                block, channels[i], layers[i], stride=2, groups=gs[i],
                se_reduction=se_reduction, se_last=se_last, se_mode=se_mode,
                avg_shortcut=avg_shortcut)
            setattr(self, "layer" + str(i + 1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, groups, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        groups=groups, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                groups=groups, **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


def RegNetY_200MF(**kwargs):
    return RegNet(32, (24, 56, 152, 368), (1, 1, 4, 7), 8, 4, **kwargs)

def RegNetY_400MF(**kwargs):
    return RegNet(32, (48, 104, 208, 440), (1, 3, 6, 6), 8, 4, **kwargs)

def RegNetY_600MF(**kwargs):
    return RegNet(32, (48, 112, 256, 608), (1, 3, 7, 4), 16, 4, **kwargs)

def RegNetY_800MF(**kwargs):
    return RegNet(32, (64, 128, 320, 768), (1, 3, 8, 2), 16, 4, **kwargs)

def RegNetY_1_6GF(**kwargs):
    return RegNet(32, (48, 120, 336, 888), (2, 6, 17, 2), 24, 4, **kwargs)

def RegNetY_3_2GF(**kwargs):
    return RegNet(32, (72, 216, 576, 1512), (2, 5, 13, 1), 24, 4, **kwargs)

def RegNetY_4_0GF(**kwargs):
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)

def RegNetY_6_4GF(**kwargs):
    return RegNet(32, (144, 288, 576, 1296), (2, 7, 14, 2), 72, 4, **kwargs)

def RegNetY_8_0GF(**kwargs):
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)

def RegNetY_12GF(**kwargs):
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)

def RegNetY_16GF(**kwargs):
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)

def RegNetY_32GF(**kwargs):
    return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)
