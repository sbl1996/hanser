from hanser.models.cifar.res2net.resnet_vd import Bottle2neck
from hanser.models.imagenet.gen_resnet.resnet_vd import ResNet as GenResNet


class ResNet(GenResNet):

    def __init__(self, block, layers, base_width=26, scale=4,
                 zero_init_residual=False, avd=False,
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__(
            block, layers, num_classes, stages,
            base_width=base_width, scale=scale,
            zero_init_residual=zero_init_residual, avd=avd,
        )

def resnet50(**kwargs):
    return ResNet(Bottle2neck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottle2neck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottle2neck, [3, 8, 36, 3], **kwargs)