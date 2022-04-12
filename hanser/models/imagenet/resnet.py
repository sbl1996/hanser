from tensorflow.keras import Model

from hanser.models.layers import GlobalAvgPool, Linear, Dropout
from hanser.models.common.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetStem
from hanser.models.common.modules import make_layer
from hanser.models.utils import init_layer_ascending_drop_path


def _get_kwargs(kwargs, i):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple) and len(v) == 4:
            d[k] = v[i]
        else:
            d[k] = v
    return d


class _ResNet(Model):

    def __init__(self, stem, block, layers, num_classes=1000, channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()

        self.stem = stem
        c_in = stem.out_channels

        blocks = (block,) * 4 if not isinstance(block, tuple) else block
        for i, (block, c, n, s) in enumerate(zip(blocks, channels, layers, strides)):
            layer = make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

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


class ResNet(_ResNet):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 64, 128, 256, 512),
                 dropout=0, zero_init_residual=False, drop_path=0):
        stem_channels, *channels = channels
        stem = ResNetStem(stem_channels)
        super().__init__(stem, block, layers, num_classes, channels, dropout=dropout,
                         zero_init_residual=zero_init_residual, drop_path=drop_path)
        if drop_path:
            init_layer_ascending_drop_path(self, drop_path)



def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)