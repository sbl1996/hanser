from hanser.models.imagenet.gen_resnet.resnet_vd import ResNet as GenResNet
from hanser.models.cifar.senet.resnet_vd import BasicBlock, Bottleneck


class ResNet(GenResNet):

    def __init__(self, block, layers, num_classes=1000, stages=(64, 64, 128, 256, 512),
                 zero_init_residual=False, reduction=16):
        super().__init__(
            block, layers, num_classes, stages,
            zero_init_residual=zero_init_residual, reduction=reduction
        )


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