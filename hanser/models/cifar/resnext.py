from hanser.models.common.resnext import Bottleneck

from hanser.models.cifar.resnet import _ResNet


class ResNeXt(_ResNet):

    def __init__(self, depth, cardinality, base_width,
                 num_classes=10, channels=(64, 64, 128, 256),
                 zero_init_residual=False):
        super().__init__(depth, Bottleneck, num_classes, channels,
                         cardinality=cardinality, base_width=base_width,
                         zero_init_residual=zero_init_residual)


def resnext29_8_64d(**kwargs):
    return ResNeXt(29, cardinality=8, base_width=64, **kwargs)


def resnext29_16_64d(**kwargs):
    return ResNeXt(29, cardinality=16, base_width=64, **kwargs)