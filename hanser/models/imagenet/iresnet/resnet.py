import numpy as np
from tensorflow.keras import Sequential, Model

from hanser.models.layers import GlobalAvgPool, Linear, Dropout
from hanser.models.common.iresnet.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetStem

class LayerArgs:

    def __init__(self, args):
        self.args = args


def _get_layer_kwargs(kwargs, i):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, LayerArgs):
            d[k] = v.args[i]
        else:
            d[k] = v
    return d


def _make_layer(block, in_channels, channels, blocks, stride, return_seq=True, **kwargs):
    layers = [block(in_channels, channels, stride=stride,
                    start_block=True, **_get_layer_kwargs(kwargs, 0))]
    in_channels = channels * block.expansion
    for i in range(1, blocks - 1):
        layers.append(block(in_channels, channels, stride=1,
                            exclude_bn0=i == 1, **_get_layer_kwargs(kwargs, i)))
    layers.append(block(in_channels, channels, stride=1,
                        end_block=True, **_get_layer_kwargs(kwargs, blocks - 1)))
    if return_seq:
        layers = Sequential(layers)
    return layers


def _get_kwargs(kwargs, i, layers=None):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if len(v) == 4:
                d[k] = v[i]
            elif layers is not None and len(v) == sum(layers):
                d[k] = LayerArgs(v[sum(layers[:i]):sum(layers[:(i+1)])])
            else:
                d[k] = v
        else:
            d[k] = v
    return d


class _IResNet(Model):

    def __init__(self, stem, block, layers, num_classes=1000, channels=(64, 128, 256, 512),
                 strides=(2, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()

        self.stem = stem
        c_in = stem.out_channels

        blocks = (block,) * 4 if not isinstance(block, tuple) else block
        for i, (block, c, n, s) in enumerate(zip(blocks, channels, layers, strides)):
            layer = _make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i, layers))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class IResNet(_IResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 drop_path=0., dropout=0):
        stem_channels, *channels = channels
        stem = ResNetStem(stem_channels, pool=False)
        super().__init__(stem, block, layers, num_classes, channels,
                         pool_type='max', drop_path=drop_path,
                         dropout=dropout)


def resnet18(**kwargs):
    return IResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return IResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return IResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return IResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return IResNet(Bottleneck, [3, 8, 36, 3], **kwargs)