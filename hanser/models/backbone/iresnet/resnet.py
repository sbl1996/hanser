from tensorflow.keras import Sequential, Model

from hanser.models.imagenet.iresnet.resnet import _make_layer, _get_kwargs
from hanser.models.layers import GlobalAvgPool, Linear, Dropout


class _IResNet(Model):

    def __init__(self, stem, block, layers, num_classes=1000, channels=(64, 128, 256, 512),
                 strides=(2, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()

        self.stem = stem
        c_in = stem.out_channels

        blocks = (block,) * 4 if not isinstance(block, tuple) else block
        for i, (block, c, n, s) in enumerate(zip(blocks, channels, layers, strides)):
            layer = _make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

        self.feat_channels = [c * block.expansion for c in channels]

    def call(self, x):
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5