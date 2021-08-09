from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.resnet import _ResNet
from hanser.models.common.senet.resnet_vd import BasicBlock, Bottleneck


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 reduction=16, se_mode=0, drop_path=0, dropout=0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, dropout=dropout,
                         reduction=reduction, se_mode=se_mode, drop_path=drop_path)

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