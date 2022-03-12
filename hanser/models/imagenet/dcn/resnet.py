from hanser.models.common.dcn.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetStem
from hanser.models.imagenet.resnet import _ResNet


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 dropout=0, zero_init_residual=False, dcn=(False, True, True, True)):
        stem_channels, *channels = channels
        stem = ResNetStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         dropout=dropout, zero_init_residual=zero_init_residual, dcn=dcn)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
