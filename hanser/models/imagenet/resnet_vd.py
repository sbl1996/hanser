from hanser.models.common.resnet_vd import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem, ResNetStem
from hanser.models.imagenet.resnet import _ResNet


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 deep_stem=True, maxpool=True, dropout=0.0, zero_init_residual=False):
        stem_channels, *channels = channels
        Stem = ResNetvdStem if deep_stem else ResNetStem
        stem = Stem(stem_channels, pool=maxpool)
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1 if maxpool else 2, 2, 2, 2),
                         dropout=dropout, zero_init_residual=zero_init_residual)

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# 539.3 34.5M 6.08G
def resnet74(**kwargs):
    return ResNet(Bottleneck, [3, 4, 14, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
