from hanser.models.common.resnetpp.group import Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 128, 256, 512, 1024),
                 group_width=32, inv_expansion=2.0, drop_path=0, dropout=0,
                 se_reduction=(1, 2, 2, 2), se_mode=0):
        stem_channels, *channels = channels
        stem = SpaceToDepthStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1, 2, 2, 2), dropout=dropout, drop_path=drop_path,
                         group_width=group_width, inv_expansion=inv_expansion,
                         se_last=True, se_reduction=se_reduction, se_mode=se_mode)


def resnet_s_32d(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], group_width=32, **kwargs)


def resnet_s_64d(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], group_width=64, **kwargs)


def resnet_s_128d(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], group_width=128, **kwargs)