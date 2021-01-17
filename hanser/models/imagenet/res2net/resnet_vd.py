from tensorflow.keras import Sequential, Model

from hanser.models.layers import Conv2d, GlobalAvgPool, Linear, Pool2d
from hanser.models.cifar.res2net.resnet_vd import Bottle2neck


class ResNet(Model):

    def __init__(self, block, layers, base_width=26, scale=4,
                 erase_relu=False, zero_init_residual=False,
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__()
        self.stages = stages

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
            base_width=base_width, scale=scale, erase_relu=erase_relu,
            zero_init_residual=zero_init_residual)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, scale=scale, erase_relu=erase_relu,
            zero_init_residual=zero_init_residual)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, scale=scale, erase_relu=erase_relu,
            zero_init_residual=zero_init_residual)
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=2,
            base_width=base_width, scale=scale, erase_relu=erase_relu,
            zero_init_residual=zero_init_residual)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, start_block=True,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                **kwargs))
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


def resnet50(**kwargs):
    return ResNet(Bottle2neck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottle2neck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottle2neck, [3, 8, 36, 3], **kwargs)