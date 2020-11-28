import math
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Linear

class SKLayer(Layer):
    def __init__(self, in_channels, num_paths, reduction):
        super().__init__()
        self.num_paths = num_paths
        channels = min(max(in_channels // reduction, 32), in_channels)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.fc = Sequential([
            Conv2d(in_channels, channels, 1, norm='def', act='def'),
            Conv2d(channels, in_channels * num_paths, 1),
        ])

    def call(self, x):
        assert x.shape[3] == self.num_paths
        x = tf.reduce_sum(x, axis=3)
        x = self.pool(x)
        x = self.fc(x)
        x = tf.reshape(x, [tf.shape(x)[0], 1, 1, self.num_paths, x.shape[-1] // self.num_paths])
        x = tf.nn.softmax(x, axis=3)
        return x


class SKConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_sizes, stride, groups, dilations,
                 reduction, norm, act):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.paths = [
            Conv2d(in_channels, out_channels, kernel_size=k, stride=stride,
                   groups=groups, dilation=d, norm=norm, act=act)
            for k, d in zip(kernel_sizes, dilations)
        ]
        self.sk = SKLayer(out_channels, len(kernel_sizes), reduction)

    def call(self, x):
        xs = [op(x) for op in self.paths]
        x = tf.stack(xs, axis=3)
        s = self.sk(x)
        x = x * s
        x = tf.reduce_sum(x, axis=3)
        return x


class Bottleneck(Layer):

    def __init__(self, in_channels, channels, stride, cardinality, base_width, reduction):
        super().__init__()
        out_channels = channels * 4

        D = math.floor(channels * (base_width / 64))
        C = cardinality

        self.conv1 = Conv2d(in_channels, D * C, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = SKConv(D * C, D * C, kernel_sizes=(3, 1), stride=stride, groups=cardinality,
                            dilations=(1, 1), reduction=reduction, norm='def', act='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1,
                            norm='def')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm='def') if in_channels != out_channels else Identity()
        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNeXt(Model):
    stages = [64, 64, 128, 256]

    def __init__(self, depth, cardinality, base_width, reduction=16, num_classes=10):
        super().__init__()
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')

        self.layer1 = self._make_layer(
            self.stages[0], self.stages[1], layers[0], 1,
            cardinality, base_width, reduction)
        self.layer2 = self._make_layer(
            self.stages[1], self.stages[2], layers[1], 2,
            cardinality, base_width, reduction)
        self.layer3 = self._make_layer(
            self.stages[2], self.stages[3], layers[2], 2,
            cardinality, base_width, reduction)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3], num_classes)

    def _make_layer(self, in_channels, channels, blocks, stride,cardinality, base_width, reduction):
        layers = [Bottleneck(in_channels, channels, stride,cardinality, base_width, reduction)]
        out_channels = channels * 4
        for i in range(1, blocks):
            layers.append(Bottleneck(out_channels, channels, 1, cardinality, base_width, reduction))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


