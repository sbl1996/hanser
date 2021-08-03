from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dropout, Layer

from hanser.models.layers import GlobalAvgPool, Linear, Conv2d, Act
from hanser.models.attention import SELayer
from hanser.models.imagenet.common import  SpaceToDepthStem, get_shortcut_vd

from hanser.models.modules import AntiAliasing


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride, reduction=4,
                 zero_init_residual=True, act='leaky_relu'):
        super().__init__()
        out_channels = channels * self.expansion
        if stride == 1:
            self.conv1 = Conv2d(
                in_channels, out_channels, kernel_size=3, norm='def', act=act)
        else:
            self.conv1 = Sequential([
                Conv2d(in_channels, out_channels, kernel_size=3, norm='def', act=act),
                AntiAliasing()
            ])
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='def', gamma_init='zeros' if zero_init_residual else 'ones')
        self.se = SELayer(out_channels, reduction=reduction, min_se_channels=64)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.act = Act('relu')

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = x + identity
        x = self.act(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, use_se=True, reduction=8,
                 zero_init_residual=True, act='leaky_relu'):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act=act)
        if stride == 1:
            self.conv2 = Conv2d(channels, channels, kernel_size=3, norm='def', act=act)
        else:
            self.conv2 = Sequential([
                Conv2d(channels, channels, kernel_size=3, norm='def', act=act),
                AntiAliasing()
            ])

        se_channels = max(out_channels // reduction, 64)
        self.se = SELayer(channels, se_channels=se_channels) if use_se else None

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='def', gamma_init='zeros' if zero_init_residual else 'ones')

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.act = Act('relu')

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class TResNet(Model):
    # TResNet use leaky_relu with 1e-3 or 1e-6. We found it the same with ReLU.
    
    def __init__(self, layers, num_classes=1000, stages=(64, 64, 128, 256, 512),
                 zero_init_residual=True, dropout=0.0, act='leaky_relu'):
        super().__init__()
        self.stages = stages

        self.block_1 = BasicBlock
        self.block_2 = Bottleneck

        self.stem = SpaceToDepthStem(channels=stages[0])
        self.in_channels = self.stages[0]


        self.layer1 = self._make_layer(
            BasicBlock, self.stages[1], layers[0], stride=1,
            reduction=4, zero_init_residual=zero_init_residual,
            act=act)
        self.layer2 = self._make_layer(
            BasicBlock, self.stages[2], layers[1], stride=2,
            reduction=4, zero_init_residual=zero_init_residual,
            act=act)
        self.layer3 = self._make_layer(
            Bottleneck, self.stages[3], layers[2], stride=2,
            reduction=8, zero_init_residual=zero_init_residual,
            act=act)
        self.layer4 = self._make_layer(
            Bottleneck, self.stages[4], layers[3], stride=2,
            use_se=False, zero_init_residual=zero_init_residual,
            act=act)

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


def TResNet_M(**kwargs):
    return TResNet([3, 4, 11, 3], stages=(64, 64, 128, 256, 512), **kwargs)


def TResNet_L(**kwargs):
    return TResNet([4, 5, 18, 3], stages=(76, 76, 152, 304, 608), **kwargs)


def TResNet_XL(**kwargs):
    return TResNet([4, 5, 24, 3], stages=(84, 84, 168, 336, 672), **kwargs)
