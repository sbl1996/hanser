from hanser.models.cifar.ppnas.resnet_vd import Bottleneck
from hanser.models.imagenet.gen_resnet.resnet_vd import ResNet as GenResNet


class ResNet(GenResNet):

    def __init__(self, genotype, block, layers, base_width=24, splits=4, zero_init_residual=False,
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__(
            block, layers, num_classes, stages,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype
        )

def resnet50(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 8, 36, 3], **kwargs)