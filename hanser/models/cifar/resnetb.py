import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Act, Conv2d, Norm, GlobalAvgPool, Linear, Identity, Pool2d


class PreActResBlock(Layer):
    def __init__(self, in_channels, out_channels, stride):
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

    def call(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + self.shortcut(identity)


class ResNet(Model):
    stages = [16, 16, 32, 64]

    def __init__(self, depth=32, k=4, deep_stem=True, num_classes=10):
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

        self.layer1 = self._make_layer(stages[0], stages[1], num_blocks, stride=1)
        self.layer2 = self._make_layer(stages[1], stages[2], num_blocks, stride=2)
        self.layer3 = self._make_layer(stages[2], stages[3], num_blocks, stride=2)

        self.norm = Norm(stages[3])
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [PreActResBlock(in_channels, out_channels, stride=stride)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, stride=1))
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


class DirectAddFuse(Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, residual = inputs
        xo = x + residual
        return xo


class AFF(Layer):

    def __init__(self, in_channels, out_channels=64, r=4):
        super().__init__()
        inter_channels = int(out_channels // r)

        self.local_att = Sequential([
            Conv2d(in_channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, out_channels, kernel_size=1, norm='def')
        ])

        self.global_att = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(in_channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, out_channels, kernel_size=1, norm='def')
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

    def __init__(self, in_channels, out_channels=64, r=4):
        super().__init__()
        inter_channels = int(out_channels // r)

        self.local_att = Sequential([
            Conv2d(in_channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, out_channels, kernel_size=1, norm='def')
        ])

        self.global_att = Sequential([
            GlobalAvgPool(keep_dim=True),
            Conv2d(in_channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, out_channels, kernel_size=1, norm='def')
        ])

        self.local_att2 = Sequential([
            Conv2d(out_channels, inter_channels, kernel_size=1, norm='def', act='def'),
            Conv2d(inter_channels, out_channels, kernel_size=1, norm='def')
        ])

        self.local_att2 = nn.HybridSequential(prefix='local_att2')
        self.local_att2.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
        self.local_att2.add(nn.BatchNorm())
        self.local_att2.add(nn.Activation('relu'))
        self.local_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
        self.local_att2.add(nn.BatchNorm())

        self.global_att2 = nn.HybridSequential(prefix='global_att2')
        self.global_att2.add(nn.GlobalAvgPool2D())
        self.global_att2.add(nn.Conv2D(inter_channels, kernel_size=1, strides=1, padding=0))
        self.global_att2.add(nn.BatchNorm())
        self.global_att2.add(nn.Activation('relu'))
        self.global_att2.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
        self.global_att2.add(nn.BatchNorm())

        self.sig1 = nn.Activation('sigmoid')
        self.sig2 = nn.Activation('sigmoid')

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
        xlg2 = F.broadcast_add(xl2, xg2)
        wei2 = self.sig2(xlg2)
        xo = F.broadcast_mul(x, wei2) + F.broadcast_mul(residual, 1 - wei2)

        return xo