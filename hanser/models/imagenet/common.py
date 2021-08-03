from hanser.models.modules import SpaceToDepth
from tensorflow.keras import Sequential
from hanser.models.layers import Conv2d, Pool2d, Identity


def SimpleStem(channels=64):
    return Conv2d(3, channels, kernel_size=3, stride=2, norm='def', act='def')


def ResNetStem(channels=64, pool=True):
    layers = [
        Conv2d(3, channels, kernel_size=7, stride=2, norm='def', act='def'),
    ]
    if pool:
        layers.append(Pool2d(kernel_size=3, stride=2, type='max'))
    return Sequential(layers)


def ResNetvdStem(channels=64, pool=True):
    layers = [
        Conv2d(3, channels // 2, kernel_size=3, stride=2,
               norm='def', act='def'),
        Conv2d(channels // 2, channels // 2, kernel_size=3,
               norm='def', act='def'),
        Conv2d(channels // 2, channels, kernel_size=3,
               norm='def', act='def'),
    ]
    if pool:
        layers.append(Pool2d(kernel_size=3, stride=2, type='max'))
    return Sequential(layers)


def SpaceToDepthStem(channels=64):
    layers = [
        SpaceToDepth(4),
        Conv2d(3 * 16, channels, 3, stride=1, norm='def', act='def')
    ]
    return Sequential(layers)


def get_shortcut_vd(in_channels, out_channels, stride):
    if stride != 1 or in_channels != out_channels:
        shortcut = []
        if stride != 1:
            shortcut.append(Pool2d(2, 2, type='avg'))
        shortcut.append(
            Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
        shortcut = Sequential(shortcut)
    else:
        shortcut = Identity()
    return shortcut