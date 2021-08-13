from tensorflow.keras import Model

from hanser.models.layers import GlobalAvgPool, Linear, Dropout
from hanser.models.common.preactresnet import BasicBlock, Bottleneck
from hanser.models.common.modules import make_layer
from hanser.models.layers import NormAct
from hanser.models.imagenet.stem import ResNetStem


class _ResNet(Model):

    def __init__(self, stem, block, layers, num_classes=1000, channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()
        self.stem = stem
        c_in = stem.out_channels

        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(block, c_in, c, n, s, **kwargs)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.norm_act = NormAct(c_in)
        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.norm_act(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 dropout=0):
        stem_channels, *channels = channels
        stem = ResNetStem(stem_channels, norm_act=False)
        super().__init__(stem, block, layers, num_classes, channels,
                         dropout=dropout)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)