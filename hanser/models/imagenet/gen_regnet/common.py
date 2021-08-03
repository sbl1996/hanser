from tensorflow.keras import Model, Sequential

from hanser.models.layers import GlobalAvgPool, Linear
from hanser.models.imagenet.stem import SimpleStem


class GenRegNet(Model):

    def __init__(self, block, stem_channels, stages, layers, channels_per_group, num_classes=1000, **kwargs):
        super().__init__()

        self.stem = SimpleStem(stem_channels)
        self.in_channels = stem_channels
        gs = [c // channels_per_group for c in stages]

        self.stage1 = self._make_layer(
            block, stages[0], layers[0], stride=2, groups=gs[0], **kwargs)
        self.stage2 = self._make_layer(
            block, stages[1], layers[1], stride=2, groups=gs[1], **kwargs)
        self.stage3 = self._make_layer(
            block, stages[2], layers[2], stride=2, groups=gs[2], **kwargs)
        self.stage4 = self._make_layer(
            block, stages[3], layers[3], stride=2, groups=gs[3], **kwargs)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x