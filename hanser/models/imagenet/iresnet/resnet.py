from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dropout

from hanser.models.layers import GlobalAvgPool, Linear
from hanser.models.cifar.iresnet.resnet import BasicBlock, Bottleneck
from hanser.models.imagenet.common import ResNetStem, ResNetvdStem


class ResNet(Model):

    def __init__(self, block, layers, num_classes=1000, stages=(64, 64, 128, 256, 512),
                 deep_stem=False, pool_type='max', drop_path=0, dropout=0):
        super().__init__()
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0], pool=False) if deep_stem \
            else ResNetStem(self.stages[0], pool=False)
        self.in_channels = self.stages[0]

        for i in range(4):
            layer = self._make_layer(
                block, self.stages[i + 1], layers[i], stride=2,
                pool_type=pool_type, drop_path=drop_path)
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride,
                        start_block=True, **kwargs)]
        self.in_channels = channels * block.expansion

        for i in range(1, blocks - 1):
            layers.append(block(self.in_channels, channels, stride=1,
                                exclude_bn0=i == 1, **kwargs))

        layers.append(block(self.in_channels, channels, stride=1,
                            end_block=True, **kwargs))

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