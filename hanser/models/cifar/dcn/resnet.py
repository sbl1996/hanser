from hanser.models.common.dcn.resnet import Bottleneck
from hanser.models.cifar.resnet import _ResNet


class ResNet(_ResNet):

    def __init__(self, depth, block, num_classes=10, channels=(16, 16, 32, 64),
                 zero_init_residual=False, dcn=True):
        super().__init__(depth, block, num_classes, channels,
                         zero_init_residual=zero_init_residual, dcn=dcn)


def resnet110(**kwargs):
    return ResNet(depth=110, block=Bottleneck, **kwargs)