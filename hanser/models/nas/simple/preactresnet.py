from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import NormAct, Conv2d, GlobalAvgPool, Linear
from hanser.models.modules import DropPath, Identity
from hanser.models.common.modules import make_layer


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, drop_path=0):
        super().__init__()
        out_channels = channels * self.expansion

        self.norm_act1 = NormAct(in_channels)
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1)

        self.norm_act2 = NormAct(channels)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride)

        self.norm_act3 = NormAct(channels)
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def call(self, x):
        shortcut = x
        x = self.norm_act1(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm_act2(x)
        x = self.conv2(x)
        x = self.norm_act3(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        return x + shortcut


class _ResNet(Model):

    def __init__(self, depth, block, num_classes=100, channels=(16, 16, 32, 64), drop_path=0):
        super().__init__()
        layers = [(depth - 2) // 9] * 3

        stem_channels, *channels = channels

        self.stem = Conv2d(3, stem_channels, kernel_size=3)
        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(block, c_in, c, n, s,
                               drop_path=drop_path)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.norm_act = NormAct(c_in)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm_act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNet(_ResNet):

    def __init__(self, depth, k=1, block=Bottleneck, num_classes=100, channels=(16, 16, 32, 64), drop_path=0):
        channels = (channels[0],) + tuple(c * k for c in channels[1:])
        super().__init__(depth, block, num_classes, channels, drop_path)

def resnet110(**kwargs):
    return ResNet(depth=110, **kwargs)