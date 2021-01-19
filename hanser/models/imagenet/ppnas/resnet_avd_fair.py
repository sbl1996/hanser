from tensorflow.keras import Model, Sequential
from hanser.models.layers import Conv2d, Linear, GlobalAvgPool, Pool2d

from hanser.models.cifar.ppnas.resnet_avd_fair import Bottleneck


class ResNet(Model):

    def __init__(self, genotype, block, layers, base_width=26, splits=4, zero_init_residual=False,
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        genotype = genotype.normal

        self.stem = Sequential([
            Conv2d(3, self.stages[0] // 2, kernel_size=3, stride=2,
                   norm='def', act='def'),
            Conv2d(self.stages[0] // 2, self.stages[0] // 2, kernel_size=3,
                   norm='def', act='def'),
            Conv2d(self.stages[0] // 2, self.stages[0], kernel_size=3,
                   norm='def', act='def'),
        ])
        self.maxpool = Pool2d(kernel_size=3, stride=2, type='max')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[0])
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[1])
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[2])
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=2,
            base_width=base_width, splits=splits,
            zero_init_residual=zero_init_residual, genotype=genotype[3])

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def resnet50(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 8, 36, 3], **kwargs)