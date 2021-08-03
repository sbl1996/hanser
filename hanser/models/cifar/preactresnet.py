from tensorflow.keras import Model

from hanser.models.layers import NormAct, Conv2d, GlobalAvgPool, Linear
from hanser.models.common.modules import make_layer
from hanser.models.common.preactresnet import BasicBlock


class _ResNet(Model):

    def __init__(self, depth, block, num_classes=10, channels=(16, 16, 32, 64), dropout=0):
        super().__init__()
        layers = [(depth - 4) // 6] * 3

        stem_channels, *channels = channels

        self.stem = Conv2d(3, stem_channels, kernel_size=3)
        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(block, c_in, c, n, s,
                               dropout=dropout)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.norm_act = NormAct(c_in)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm_act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNet(_ResNet):

    def __init__(self, depth, k, block=BasicBlock, num_classes=10, channels=(16, 16, 32, 64), dropout=0):
        channels = (channels[0],) + tuple(c * k for c in channels[1:])
        super().__init__(depth, block, num_classes, channels, dropout)


def WRN_16_8(**kwargs):
    return ResNet(depth=16, k=8, block=BasicBlock, **kwargs)


def WRN_28_10(**kwargs):
    return ResNet(depth=28, k=10, block=BasicBlock, **kwargs)