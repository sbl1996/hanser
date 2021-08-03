from hanser.models.common.preactresnet_vd import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.preactresnet import _ResNet


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 dropout=0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels, norm_act=False)
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