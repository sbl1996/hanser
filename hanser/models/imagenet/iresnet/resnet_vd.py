from hanser.models.cifar.iresnet.resnet_vd import BasicBlock, Bottleneck
from hanser.models.imagenet.iresnet.resnet import ResNet


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], deep_stem=True, pool_type='avg', **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], deep_stem=True, pool_type='avg', **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, pool_type='avg', **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], deep_stem=True, pool_type='avg', **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], deep_stem=True, pool_type='avg', **kwargs)