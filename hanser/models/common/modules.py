from tensorflow.keras import Sequential
from hanser.models.layers import Pool2d, Conv2d, Identity


def get_shortcut_vd(in_channels, out_channels, stride,
                    norm='def', identity=True, pool_type='avg'):
    if stride != 1 or in_channels != out_channels:
        shortcut = []
        if stride != 1:
            if pool_type == 'max':
                pool = Pool2d(3, 2, type='max')
            elif pool_type == 'avg':
                pool = Pool2d(2, 2, type='avg')
            # noinspection PyUnboundLocalVariable
            shortcut.append(pool)
        shortcut.append(
            Conv2d(in_channels, out_channels, kernel_size=1, norm=norm))
        shortcut = Sequential(shortcut)
    else:
        shortcut = Identity() if identity else None
    return shortcut


def get_shortcut(in_channels, out_channels, stride, norm='def', identity=True):
    if in_channels != out_channels or stride != 1:
        shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, norm=norm)
    else:
        shortcut = Identity() if identity else None
    return shortcut


def make_layer(block, in_channels, channels, blocks, stride=1, **kwargs):
    layers = [block(in_channels, channels, stride=stride, **kwargs)]
    in_channels = channels * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, channels, stride=1, **kwargs))
    return Sequential(layers)