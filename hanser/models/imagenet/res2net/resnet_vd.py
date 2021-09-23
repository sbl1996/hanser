from hanser.models.common.res2net.resnet_vd import Bottle2neck
from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.resnet import _ResNet


class Res2Net(_ResNet):

    def __init__(self, block, layers, base_width=26, scale=4,
                 num_classes=1000, channels=(64, 64, 128, 256, 512),
                 zero_init_residual=False, dropout=0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, dropout=dropout,
                         base_width=base_width, scale=scale,
                         zero_init_residual=zero_init_residual)


def resnet50(**kwargs):
    return Res2Net(Bottle2neck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return Res2Net(Bottle2neck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return Res2Net(Bottle2neck, [3, 8, 36, 3], **kwargs)