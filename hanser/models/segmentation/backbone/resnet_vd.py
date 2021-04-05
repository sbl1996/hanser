from hanser.models.cifar.resnet_vd import Bottleneck
from hanser.models.backbone.resnet_vd import ResNet

def resnet50(output_stride=16, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], output_stride=output_stride, **kwargs)

def resnet101(output_stride=16, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride, **kwargs)

def resnet152(output_stride=16, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], output_stride=output_stride, **kwargs)