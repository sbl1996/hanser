from hanser.models.common.preactresnet_vd import BasicBlock
from hanser.models.cifar.preactresnet import _ResNet


class ResNet(_ResNet):

    def __init__(self, depth, k, block=BasicBlock, num_classes=10, channels=(16, 16, 32, 64), dropout=0):
        channels = (channels[0],) + tuple(c * k for c in channels[1:])
        super().__init__(depth, block, num_classes, channels, dropout)


def WRN_16_8(**kwargs):
    return ResNet(depth=16, k=8, block=BasicBlock, **kwargs)


def WRN_28_10(**kwargs):
    return ResNet(depth=28, k=10, block=BasicBlock, **kwargs)