from hanser.models.common.resnetpp.nas import Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem


class IResNet(_IResNet):

    def __init__(self, genotype, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 base_width=26, scale=4, drop_path=0, dropout=0, se_reduction=(4, 8, 8, 8), se_mode=0):
        stem_channels, *channels = channels
        stem = SpaceToDepthStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, strides=(1, 2, 2, 2),
                         dropout=dropout, drop_path=drop_path, base_width=base_width, scale=scale,
                         se_last=True, se_reduction=se_reduction, se_mode=se_mode, genotype=genotype)

def resnet_m(genotype, **kwargs):
    return IResNet(genotype, Bottleneck, [3, 4, 8, 3], **kwargs)