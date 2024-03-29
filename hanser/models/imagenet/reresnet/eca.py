from hanser.models.common.reresnet.eca import Bottleneck
from hanser.models.imagenet.iresnet.resnet import _IResNet
from hanser.models.imagenet.stem import SpaceToDepthStem


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 drop_path=0, dropout=0):
        stem_channels, *channels = channels
        stem = SpaceToDepthStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels,
                         strides=(1, 2, 2, 2), dropout=dropout, drop_path=drop_path)

def re_resnet_s(**kwargs):
    return IResNet(Bottleneck, [3, 4, 8, 3], **kwargs)
