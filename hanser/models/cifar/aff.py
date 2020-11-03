import math
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Act, Conv2d, Norm, GlobalAvgPool, Linear, Identity, Pool2d


class PreActResBlock(Layer):
    def __init__(self, in_channels, out_channels, stride, askc_type='DirectAdd'):
        super().__init__()
        self.norm1 = Norm(in_channels)
        self.act1 = Act()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.norm2 = Norm(out_channels)
        self.act2 = Act()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, bias=False)

        if in_channels != out_channels or stride == 2:
            if stride == 2:
                self.shortcut = Sequential([
                    Pool2d(2, 2, type='avg'),
                    Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                ])
            else:
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = Identity()

        if askc_type == 'DirectAdd':
            self.attention = DirectAddFuse()
        elif askc_type == 'ResGlobLocaforGlobLocaCha':
            self.attention = iAFF(out_channels)
        elif askc_type == 'ASKCFuse':
            self.attention = AFF(out_channels)
        else:
            raise ValueError('Unknown askc_type')

    def call(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.attention((x, self.shortcut(identity)))
        return x


class AFFResNeXtBlock(Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, cardinality, bottleneck_width, stride, use_se, askc_type):
        super().__init__()
        r = 4 * self.expansion
        channels = out_channels // self.expansion
        D = int(math.floor(channels * bottleneck_width / 64))
        group_width = cardinality * D

        self.body = Sequential([
            Conv2d(in_channels, group_width, kernel_size=1, norm='def', act='def'),
            Conv2d(group_width, group_width, kernel_size=3, groups=cardinality,
                   norm='def', act='def'),
            Conv2d(group_width, out_channels, kernel_size=1, norm='def', act='def'),
        ])

        if use_se:
            self.se = Sequential([
                GlobalAvgPool(keep_dim=True),
                Conv2d(out_channels, out_channels // r, kernel_size=1, act='def'),
                Conv2d(out_channels // r, out_channels, kernel_size=1, act='sigmoid'),
            ])
        else:
            self.se = None

        if in_channels != out_channels or stride == 2:
            if stride == 2:
                self.shortcut = Sequential([
                    Pool2d(2, 2, type='avg'),
                    Conv2d(in_channels, out_channels, kernel_size=1, norm='def'),
                ])
            else:
                self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, norm='def')
        else:
            self.shortcut = Identity()

        if askc_type == 'DirectAdd':
            self.attention = DirectAddFuse()
        elif askc_type == 'ResGlobLocaforGlobLocaCha':
            self.attention = iAFF(out_channels, r=r)
        elif askc_type == 'ASKCFuse':
            self.attention = AFF(out_channels, r=r)
        else:
            raise ValueError('Unknown askc_type')

        self.post_activ = Act()

    def call(self, x):
        residual = x
        x = self.body(x)

        if self.se:
            s = self.se(x)
            x = x * s

        x = self.attention((x, residual))
        x = self.post_activ(x)
        return x


class ResNet(Model):
    stages = [16, 16, 32, 64]

    def __init__(self, depth=32, k=4, deep_stem=False, askc_type='DirectAdd', num_classes=10):
        super().__init__()
        num_blocks = (depth - 2) // 6
        stages = [c * k for c in self.stages]
        if not deep_stem:
            self.stem = Conv2d(3, stages[0], kernel_size=3, bias=False)
        else:
            stem_channels = stages[0] // 2
            self.stem = Sequential([
                Conv2d(3, stem_channels, kernel_size=3, norm='def', act='def'),
                Conv2d(stem_channels, stem_channels, kernel_size=3, norm='def', act='def'),
                Conv2d(stem_channels, stages[0], kernel_size=3, bias=False)
            ])

        self.layer1 = self._make_layer(stages[0], stages[1], num_blocks, stride=1, askc_type=askc_type)
        self.layer2 = self._make_layer(stages[1], stages[2], num_blocks, stride=2, askc_type=askc_type)
        self.layer3 = self._make_layer(stages[2], stages[3], num_blocks, stride=2, askc_type=askc_type)

        self.norm = Norm(stages[3])
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, askc_type):
        layers = [PreActResBlock(in_channels, out_channels, stride=stride, askc_type=askc_type)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, stride=1, askc_type=askc_type))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNeXt(Model):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, cardinality, bottleneck_width, use_se=False, askc_type='DirectAdd', num_classes=10):
        super().__init__()
        num_blocks = (depth - 2) // 9
        stages = [c * k * AFFResNeXtBlock.expansion for c in self.stages]
        self.stem = Conv2d(3, stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(stages[0], stages[1], num_blocks, stride=1,
                                       cardinality=cardinality, bottleneck_width=bottleneck_width,
                                       use_se=use_se, askc_type=askc_type)
        self.layer2 = self._make_layer(stages[1], stages[2], num_blocks, stride=2,
                                       cardinality=cardinality, bottleneck_width=bottleneck_width,
                                       use_se=use_se, askc_type=askc_type)
        self.layer3 = self._make_layer(stages[2], stages[3], num_blocks, stride=2,
                                       cardinality=cardinality, bottleneck_width=bottleneck_width,
                                       use_se=use_se, askc_type=askc_type)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride,
                    cardinality, bottleneck_width, use_se, askc_type):
        layers = [AFFResNeXtBlock(in_channels, out_channels, stride=stride,
                                  cardinality=cardinality, bottleneck_width=bottleneck_width,
                                  use_se=use_se, askc_type=askc_type)]
        for i in range(1, blocks):
            layers.append(
                AFFResNeXtBlock(out_channels, out_channels, stride=1,
                                cardinality=cardinality, bottleneck_width=bottleneck_width,
                                use_se=use_se, askc_type=askc_type))
            return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


class DirectAddFuse(Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, residual = inputs
        xo = x + residual
        return xo


class AFF(Layer):

    def __init__(self, channels, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        self.local_att = Sequential([
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

        self.global_att = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

    def call(self, inputs):
        x, residual = inputs
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        s = tf.sigmoid(xlg)

        xo = 2 * (x * s + residual * (1 - s))
        return xo


class iAFF(Layer):

    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        self.local_att = Sequential([
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

        self.global_att = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

        self.local_att2 = Sequential([
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

        self.global_att2 = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, channels, kernel_size=1, norm='def')
        ])

    def call(self, inputs):
        x, residual = inputs

        xa = x + residual

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        s = tf.sigmoid(xlg)

        xi = x * s + residual * (1 - s)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        s2 = tf.sigmoid(xlg2)

        xo = x * s2 + residual * (1 - s2)

        return xo
