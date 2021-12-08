import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.modules import DropPath
from hanser.models.layers import Conv2d, Act, Identity, Norm, NormAct, Pool2d
from hanser.models.attention import SELayer, ECALayer
from hanser.models.modules import PadChannel

from hanser.models.layers import GlobalAvgPool, Linear, Dropout
from hanser.models.imagenet.stem import SpaceToDepthStem, ResNetvdStem


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg'))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels)))
        super().__init__(layers)



class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 se_reduction=4, se_last=False, se_mode=0, eca=False,
                 anti_alias=False, avd=False, drop_path=0):
        super().__init__()
        self.se = None
        self.eca = None
        self.se_last = se_last

        out_channels = channels * self.expansion

        if not start_block:
            if exclude_bn0:
                self.act0 = Act()
            else:
                self.norm_act0 = NormAct(in_channels)

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', anti_alias=anti_alias, avd=avd, avd_first=False)

        if se_reduction and not self.se_last:
            self.se = SELayer(channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if se_reduction and self.se_last:
            self.se = SELayer(out_channels, se_channels=out_channels // se_reduction, mode=se_mode)

        if eca:
            self.eca = ECALayer(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.norm_act3 = NormAct(out_channels)

        self.shortcut = Shortcut(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x):
        identity = self.shortcut(x)

        if not self.start_block:
            if self.exclude_bn0:
                x = self.act0(x)
            else:
                x = self.norm_act0(x)

        x = self.conv1(x)

        x = self.conv2(x)
        if self.se is not None and not self.se_last:
            x = self.se(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        if self.se is not None and self.se_last:
            x = self.se(x)

        if self.eca is not None:
            x = self.eca(x)

        x = self.drop_path(x)
        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x


def _get_kwargs(kwargs, i):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if len(v) == 4:
                d[k] = v[i]
            else:
                d[k] = v
        else:
            d[k] = v
    return d


class ResNet(Model):

    def __init__(self, stem, block, layers, alpha, num_classes=1000,
                 strides=(2, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()

        self.stem = stem
        c_in = stem.out_channels

        channels = np.linspace(0, alpha, sum(layers)) + c_in
        channels = (np.round(channels / 8) * 8).astype(np.int)
        k = 0

        for i, (n, s) in enumerate(zip(layers, strides)):
            ls = []
            for j in range(n):
                ls.append(
                    block(c_in, channels[k], stride=s if j == 0 else 1, start_block=j == 0,
                          exclude_bn0=j == 1, end_block=j == n-1, **_get_kwargs(kwargs, i)))
                c_in = channels[k] * block.expansion
                k += 1
            layer = Sequential(ls)
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

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


class ReResNet(ResNet):

    def __init__(self, layers, num_classes=1000, alpha=360, pool=False,
                 se_reduction=(0, 0, 0, 0), se_mode=0, se_last=True, eca=False, strides=(2, 2, 2, 2),
                 anti_alias=False, avd=False, light_stem=False):
        stem_channels = 64
        if light_stem:
            stem = SpaceToDepthStem(stem_channels)
        else:
            stem = ResNetvdStem(stem_channels, pool=pool)
        super().__init__(stem, Bottleneck, layers, alpha, num_classes,
                         strides=strides, anti_alias=anti_alias, avd=avd,
                         se_mode=se_mode, se_last=se_last, se_reduction=se_reduction, eca=eca)

# 21.5M 4.66G 1025
def net1(**kwargs):
    return ReResNet(layers=(2, 4, 7, 3), alpha=320, anti_alias=(False, True, True, True),
                    light_stem=False, strides=(2, 2, 2, 2), se_reduction=(4, 8, 8, 8), **kwargs)
