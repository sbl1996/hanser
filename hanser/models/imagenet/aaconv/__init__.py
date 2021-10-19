from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, NormAct
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.common.aaconv import AAConv
from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.resnet import _ResNet


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, use_aa, n_heads=8, k=0.25, v=0.25):
        super().__init__()
        dk = int(k * channels)
        dv = int(v * channels)

        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        if use_aa:
            self.conv2 = Sequential([
                AAConv(channels, channels, kernel_size=3, stride=stride,
                       n_heads=n_heads, dk=dk, dv=dv),
                NormAct(channels),
            ])
        else:
            self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                                norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1, norm='def')

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
                 dropout=0.0, k=0.25, v=0.25):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1, 2, 2, 2), dropout=dropout,
                         use_aa=(False, True, True, True), k=k, v=v)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)