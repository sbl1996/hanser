from hanser.models.cifar.ppnas.resnet_avd_fair import Bottleneck
from hanser.models.backbone.resnet_avd_fair import ResNet


def resnet50(genotype, output_stride=16, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 6, 3], output_stride=output_stride, **kwargs)

def resnet101(genotype, output_stride=16, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 23, 3], output_stride=output_stride, **kwargs)

def resnet152(genotype, output_stride=16, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 8, 36, 3], output_stride=output_stride, **kwargs)