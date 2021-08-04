from hanser.models.common.iresnet.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import ResNetvdStem


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 pool_type='avg', drop_path=0, anti_alias=False, dropout=0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels, pool=False)
        super().__init__(stem, block, layers, num_classes, channels,
                         pool_type=pool_type, drop_path=drop_path,
                         anti_alias=anti_alias, dropout=dropout)


def resnet18(**kwargs):
    return IResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return IResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return IResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return IResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return IResNet(Bottleneck, [3, 8, 36, 3], **kwargs)