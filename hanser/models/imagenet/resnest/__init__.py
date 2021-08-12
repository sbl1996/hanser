import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Act, Conv2d, GlobalAvgPool
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.imagenet.resnet import _ResNet
from hanser.models.imagenet.stem import ResNetvdStem


class SplAtConv2d(Layer):

    def __init__(self, in_channels, channels, kernel_size, stride=1,
                 groups=1, radix=2, reduction=4, dropblock=False,
                 norm='def', act='def', **kwargs):
        super().__init__()
        self.radix = radix
        self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride,
                           groups=radix * groups, dropblock=dropblock,
                           norm=norm, act=act, **kwargs)

        self.pool = GlobalAvgPool(keep_dim=True)
        inter_channels = max(channels * radix // reduction, 32)
        self.fc = Sequential([
            Conv2d(channels, inter_channels, 1, groups=groups, norm='def', act='def'),
            Conv2d(inter_channels, channels * radix, 1, groups=groups),
        ])
        self.rsoftmax = rSoftMax(radix, groups)

    def call(self, x):
        x = self.conv(x)

        xs = tf.split(x, self.radix, axis=-1)

        u = sum(xs)
        s = self.pool(u)
        s = self.fc(s)
        attns = self.rsoftmax(s)

        out = sum([attn[:, None, None, :] * x for (attn, x) in zip(attns, xs)])
        return out


class rSoftMax(Layer):
    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def call(self, x):
        b = tf.shape(x)[0]
        c = x.shape[-1]
        c_ = c // self.groups // self.radix

        if self.groups == 1:
            x = tf.reshape(x, (b, self.radix, c_))
            x = tf.nn.softmax(x, axis=1)
            attns = tf.unstack(x, axis=1)
            return attns
        else:
            x = tf.reshape(x, (b, self.groups, self.radix, c_))
            x = tf.nn.softmax(x, axis=2)
            attns = [tf.reshape(attn, [b, self.groups * c_])
                     for attn in tf.unstack(x, axis=2)]
            return attns


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, radix=1, groups=1,
                 base_width=64, reduction=4, avd=False, avd_first=False,
                 dropblock=False):
        super().__init__()
        out_channels = channels * self.expansion
        width = int(channels * (base_width / 64)) * groups

        self.conv1 = Conv2d(in_channels, width, kernel_size=1,
                            norm='def', act='def', dropblock=dropblock)

        self.conv2 = SplAtConv2d(width, width, 3, stride=stride, groups=groups, radix=radix,
                                 reduction=reduction, avd=avd, avd_first=avd_first,
                                 norm='def', act='def', dropblock=dropblock)

        self.conv3 = Conv2d(width, out_channels, kernel_size=1,
                            norm='def', dropblock=dropblock)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 radix=2, groups=1, base_width=64, reduction=4, avd=True, avd_first=False,
                 dropblock=False, dropout=0.0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         radix=radix, groups=groups, base_width=base_width,
                         reduction=reduction, avd=avd, avd_first=avd_first,
                         dropblock=dropblock, dropout=dropout)


def resnest50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
