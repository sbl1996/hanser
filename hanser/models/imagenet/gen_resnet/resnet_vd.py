from tensorflow.keras import Sequential, Model

from hanser.models.layers import GlobalAvgPool, Linear
from hanser.models.cifar.resnet_vd import BasicBlock, Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem


class ResNet(Model):

    def __init__(self, block, layers, num_classes=1000, stages=(64, 64, 128, 256, 512), **kwargs):
        super().__init__()
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0])
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=2, **kwargs)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


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