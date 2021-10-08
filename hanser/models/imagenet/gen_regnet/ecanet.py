from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Identity
from hanser.models.imagenet.gen_regnet.regnety import GenRegNet
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


class RegNetY(GenRegNet):

    def __init__(self, stem_channels, stages, layers, channels_per_group, num_classes=1000):
        super().__init__(Bottleneck, stem_channels, stages, layers, channels_per_group, num_classes)


def regnety_200MF(**kwargs):
    return RegNetY(32, (24, 56, 152, 368), (1, 1, 4, 7), 8, **kwargs)

def regnety_400MF(**kwargs):
    return RegNetY(32, (48, 104, 208, 440), (1, 3, 6, 6), 8, **kwargs)

def regnety_600MF(**kwargs):
    return RegNetY(32, (48, 112, 256, 608), (1, 3, 7, 4), 16, **kwargs)

def regnety_800MF(**kwargs):
    return RegNetY(32, (64, 128, 320, 768), (1, 3, 8, 2), 16, **kwargs)

def regnety_1_6GF(**kwargs):
    return RegNetY(32, (48, 120, 336, 888), (2, 6, 17, 2), 24, **kwargs)

def regnety_3_2GF(**kwargs):
    return RegNetY(32, (72, 216, 576, 1512), (2, 5, 13, 1), 24, **kwargs)

def regnety_4_0GF(**kwargs):
    return RegNetY(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, **kwargs)

def regnety_6_4GF(**kwargs):
    return RegNetY(32, (144, 288, 576, 1296), (2, 7, 14, 2), 72, **kwargs)

# def regnety_8_0GF(**kwargs):
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, **kwargs)
#
# def regnety_12GF(**kwargs):
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, **kwargs)
#
# def regnety_16GF(**kwargs):
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, **kwargs)
#
# def regnety_32GF(**kwargs):
#     return RegNet(32, (128, 192, 512, 1088), (2, 6, 12, 2), 64, **kwargs)
