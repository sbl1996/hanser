from hanser.models.cifar.res2net.resnet_vd import Bottle2neck
from hanser.models.backbone.res2net_vd import ResNet

def resnet50(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottle2neck, [3, 4, 6, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)

def resnet101(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottle2neck, [3, 4, 23, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)

def resnet152(output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    return ResNet(Bottle2neck, [3, 8, 36, 3], output_stride=output_stride, multi_grad=multi_grad, **kwargs)