from hanser.models.common.resnext_vd import Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem
from hanser.models.imagenet.resnet import _ResNet


class ResNeXt(_ResNet):

    def __init__(self, block, layers, cardinality, base_width,
                 num_classes=1000, channels=(64, 64, 128, 256, 512),
                 zero_init_residual=False, dropout=0.0):
        stem_channels, *channels = channels
        stem = ResNetvdStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, dropout=dropout,
                         cardinality=cardinality, base_width=base_width,
                         zero_init_residual=zero_init_residual)


def resnext50_32x4d(**kwargs):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)


def resnext101_32x4d(**kwargs):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)


def resnext101_64x4d(**kwargs):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], cardinality=64, base_width=4, **kwargs)
