from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.attention import SELayer
from hanser.models.layers import Conv2d, Norm, Act, Identity, GlobalAvgPool, Linear


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups, use_se):
        super().__init__()
        self.use_se = use_se

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                            norm='def', act='def')
        if self.use_se:
            self.se = SELayer(out_channels, 4)
        self.conv3 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            Norm(out_channels, gamma_init='zeros')
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels, out_channels, kernel_size=1, norm='def')
        else:
            self.shortcut = Identity()
        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class RegNet(Model):

    def __init__(self, stem_channels=32, channels_per_stage=(96, 256, 640), layers=(4, 8, 2),
                 channels_per_group=16, use_se=True, num_classes=10):
        super().__init__()
        block = Bottleneck

        self.stem = Conv2d(3, stem_channels, kernel_size=3,
                           norm='def', act='def')
        self.in_channels = stem_channels
        cs = channels_per_stage
        gs = [c // channels_per_group for c in cs]

        self.stage1 = self._make_layer(
            block, cs[0], layers[0], stride=1, groups=gs[0], use_se=use_se)
        self.stage2 = self._make_layer(
            block, cs[1], layers[1], stride=2, groups=gs[1], use_se=use_se)
        self.stage3 = self._make_layer(
            block, cs[2], layers[2], stride=2, groups=gs[2], use_se=use_se)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(cs[2], num_classes)

        self.init_weights()

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

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
