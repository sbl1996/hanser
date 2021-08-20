from tensorflow.keras import Model

from hanser.models.common.resnet import BasicBlock, Bottleneck
from hanser.models.common.modules import make_layer
from hanser.models.layers import Conv2d, GlobalAvgPool, Linear


class _ResNet(Model):

    def __init__(self, depth, block, num_classes=10,
                 channels=(16, 16, 32, 64), **kwargs):
        super().__init__()
        block_name = block.__name__
        assert 'Basic' in block_name or 'Bottle' in block_name
        if 'Basic' in block_name:
            layers = [(depth - 2) // 6] * 3
        else: # 'Bottle' in block_name
            layers = [(depth - 2) // 9] * 3

        stem_channels, *channels = channels

        self.stem = Conv2d(3, stem_channels, kernel_size=3, norm='def', act='def')
        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(
                block, c_in, c, n, s, **kwargs)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNet(_ResNet):

    def __init__(self, depth, block, num_classes=10, channels=(16, 16, 32, 64),
                 zero_init_residual=False):
        if isinstance(block, str):
            if block == 'basic':
                block = BasicBlock
            else:
                block = Bottleneck
        super().__init__(depth, block, num_classes, channels,
                         zero_init_residual=zero_init_residual)


def resnet110(**kwargs):
    return ResNet(depth=110, block=Bottleneck, **kwargs)