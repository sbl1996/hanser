from hanser.models.common.iresnet.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 drop_path=0, anti_alias=True, dropout=0, use_se=True):
        stem_channels, *channels = channels
        stem = SpaceToDepthStem(stem_channels)
        se_reduction = (4, 8, 8, 8) if use_se else None
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1, 2, 2, 2), dropout=dropout,
                         pool_type='avg', drop_path=drop_path,
                         anti_alias=anti_alias, se_reduction=se_reduction)


def resnet50(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], **kwargs)