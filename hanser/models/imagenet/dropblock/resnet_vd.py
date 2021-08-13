from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import GlobalAvgPool, Linear, Conv2d, Pool2d, Identity, Act, Dropout
from hanser.models.imagenet.stem import ResNetvdStem


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride, dropblock=True):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='def', act='def', dropblock=dropblock)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='def', dropblock=dropblock)

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1,
                       norm='def', dropblock=dropblock))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, dropblock=True):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def', dropblock=dropblock)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', dropblock=dropblock)
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='def', dropblock=dropblock)

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1,
                       norm='def', dropblock=dropblock))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNet(Model):

    def __init__(self, block, layers, maxpool=True, dropout=0.0,
                 dropblock=True, dropblock_stage=(3, 4),
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__()
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0], pool=maxpool)
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1 if maxpool else 2)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            dropblock=dropblock if 3 in dropblock_stage else False)
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=2,
            dropblock=dropblock if 4 in dropblock_stage else False)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
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
        if self.dropout is not None:
            x = self.dropout(x)
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