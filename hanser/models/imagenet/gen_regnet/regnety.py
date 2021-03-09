from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Identity, GlobalAvgPool
from hanser.models.imagenet.gen_regnet.common import GenRegNet

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

    def __init__(self, in_channels, out_channels, stride, groups, reduction, zero_init_residual):
        super().__init__()
        se_channels = in_channels // reduction

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                            norm='def', act='def')
        self.se = SELayer(out_channels, se_channels)
        self.conv3 = Sequential([
            Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            Norm(out_channels, gamma_init='zeros' if zero_init_residual else 'ones')
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
        x = self.se(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class RegNetY(GenRegNet):

    def __init__(self, stem_channels, stages, layers, channels_per_group, se_reduction,
                 zero_init_residual=True, num_classes=1000):
        super().__init__(Bottleneck, stem_channels, stages, layers, channels_per_group, num_classes,
                         reduction=se_reduction, zero_init_residual=zero_init_residual)


def regnety_200MF(**kwargs):
    return RegNetY(32, (24, 56, 152, 368), (1, 1, 4, 7), 8, 4, **kwargs)

def regnety_400MF(**kwargs):
    return RegNetY(32, (48, 104, 208, 440), (1, 3, 6, 6), 8, 4, **kwargs)

def regnety_600MF(**kwargs):
    return RegNetY(32, (48, 112, 256, 608), (1, 3, 7, 4), 16, 4, **kwargs)

def regnety_800MF(**kwargs):
    return RegNetY(32, (64, 128, 320, 768), (1, 3, 8, 2), 16, 4, **kwargs)

def regnety_1_6GF(**kwargs):
    return RegNetY(32, (48, 120, 336, 888), (2, 6, 17, 2), 24, 4, **kwargs)

def regnety_3_2GF(**kwargs):
    return RegNetY(32, (72, 216, 576, 1512), (2, 5, 13, 1), 24, 4, **kwargs)

def regnety_4_0GF(**kwargs):
    return RegNetY(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, 4, **kwargs)

def regnety_6_4GF(**kwargs):
    return RegNetY(32, (144, 288, 576, 1296), (2, 7, 14, 2), 72, 4, **kwargs)

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
