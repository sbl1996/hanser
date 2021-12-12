from hanser.models.modules import SpaceToDepth
from tensorflow.keras import Sequential
from hanser.models.layers import Conv2d, Pool2d


def SimpleStem(channels=64):
    return Conv2d(3, channels, kernel_size=3, stride=2, norm='def', act='def')


def ResNetStem(channels=64, pool=True, norm_act=True):
    layers = [
        Conv2d(3, channels, kernel_size=7, stride=2, norm='def', act='def')
        if norm_act else Conv2d(3, channels, kernel_size=7, stride=2)
    ]
    if pool:
        layers.append(Pool2d(kernel_size=3, stride=2, type='max'))
    stem = Sequential(layers)
    stem.out_channels = channels
    return stem


def ResNetvdStem(channels=64, pool=True, norm_act=True):
    layers = [
        Conv2d(3, channels // 2, kernel_size=3, stride=2,
               norm='def', act='def'),
        Conv2d(channels // 2, channels // 2, kernel_size=3,
               norm='def', act='def'),
        Conv2d(channels // 2, channels, kernel_size=3, norm='def', act='def')
        if norm_act else Conv2d(channels // 2, channels, kernel_size=3),
    ]
    if pool:
        layers.append(Pool2d(kernel_size=3, stride=2, type='max'))
    stem = Sequential(layers)
    stem.out_channels = channels
    return stem


def SpaceToDepthStem(channels=64, stride=4):
    layers = [
        SpaceToDepth(stride),
        Conv2d(3 * stride * stride, channels, 3, stride=1, norm='def', act='def')
    ]
    stem = Sequential(layers)
    stem.out_channels = channels
    return stem
