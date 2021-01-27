import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, Act, Identity, GlobalAvgPool, Pool2d
from hanser.models.imagenet.gen_regnet.regnety import GenRegNet


class SELayer(Layer):

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.conv = Conv2d(channels, channels, kernel_size=1, bias=False)

    def call(self, x):
        s = self.pool(x)
        s = tf.sigmoid(self.conv(s))
        return x * s


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups, zero_init_residual, avd):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')
        if stride != 1 and avd:
            self.conv2 = Sequential([
                Pool2d(3, stride=stride, type='avg'),
                Conv2d(in_channels, out_channels,  kernel_size=3, groups=groups,
                       norm='def', act='def')
            ])
        else:
            self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                                norm='def', act='def')
        self.se = SELayer(out_channels)
        self.conv3 = Sequential([
            Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            Norm(out_channels,  gamma_init='zeros' if zero_init_residual else 'ones')
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


class RegNetY(GenRegNet):

    def __init__(self, stem_channels, stages, layers, channels_per_group,
                 zero_init_residual=False, avd=False, num_classes=1000):
        super().__init__(Bottleneck, stem_channels, stages, layers, channels_per_group, num_classes,
                         zero_init_residual=zero_init_residual, avd=avd)


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
