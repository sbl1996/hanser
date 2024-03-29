from hanser.models.common.resnet_vd import Bottleneck
from hanser.models.backbone.resnet_vd import ResNet

def resnet50(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)

def resnet101(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)

def resnet152(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)