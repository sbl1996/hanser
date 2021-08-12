from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.resnet import _ResNet
from hanser.models.common.sknet.resnet_vd import Bottleneck


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 reduction=2, dropout=0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, dropout=dropout,
                         reduction=reduction)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)