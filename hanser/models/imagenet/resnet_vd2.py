from tensorflow.keras.layers import Layer

from hanser.models.imagenet.stem import ResNetvdStem, ResNetStem
from hanser.models.imagenet.resnet import _ResNet
from hanser.models.layers import Conv2d
from hanser.models.common.resnet_vd import Bottleneck as Bottleneck_s2


class Bottleneck_s1(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        assert stride == 1
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1, norm='def')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride, act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride):
        super().__init__()
        if in_channels != channels * self.expansion or stride == 2:
            self.layer = Bottleneck_s2(in_channels, channels, stride=2)
        else:
            self.layer = Bottleneck_s1(in_channels, channels, stride=1)

    def call(self, x):
        return self.layer(x)


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 deep_stem=True, maxpool=True, dropout=0.0):
        stem_channels, *channels = channels
        Stem = ResNetvdStem if deep_stem else ResNetStem
        stem = Stem(stem_channels, pool=maxpool)
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1 if maxpool else 2, 2, 2, 2), dropout=dropout)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)