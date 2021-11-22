import numpy as np

from hanser.models.common.reresnet import Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem, ResNetvdStem


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 drop_path=0, dropout=0, se_reduction=(4, 8, 8, 8), se_mode=0, light_stem=True):
        stem_channels, *channels = channels
        if light_stem:
            stem = SpaceToDepthStem(stem_channels)
        else:
            stem = ResNetvdStem(stem_channels)
        if drop_path:
            drop_path = tuple(np.linspace(0.0, drop_path, sum(layers)))
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1, 2, 2, 2), dropout=dropout, se_last=True,
                         drop_path=drop_path, se_reduction=se_reduction, se_mode=se_mode)

def re_resnet_s(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], **kwargs)

def re_resnet_sp(layers, **kwargs):
    return IResNet(Bottleneck, layers, **kwargs)

def re_resnet_ls(**kwargs):
    return IResNet(Bottleneck, [3, 4, 6, 3], light_stem=False, **kwargs)
