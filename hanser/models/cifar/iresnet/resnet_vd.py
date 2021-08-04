# DEPRECATED
# The original IResNet uses MaxPool in projection shortcut with downsampling.
# Therefore, these is no need to use AvgPool here.

from tensorflow.keras import Sequential, Model
from hanser.models.layers import Conv2d, GlobalAvgPool, Linear
from hanser.models.common.iresnet.resnet import BasicBlock, Bottleneck


class ResNet(Model):

    def __init__(self, depth, block='basic', dropout=0, drop_path=0, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        if block == 'basic':
            block = BasicBlock
            layers = [(depth - 2) // 6] * 3
        else:
            block = Bottleneck
            layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            dropout=dropout, drop_path=drop_path)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            dropout=dropout, drop_path=drop_path)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            dropout=dropout, drop_path=drop_path)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, start_block=True,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
