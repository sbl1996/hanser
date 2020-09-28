import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer

from hanser.models.layers import Act, Conv2d, Norm, GlobalAvgPool, Linear, Identity


class PreActResBlock(Layer):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.norm1 = Norm(in_channels)
        self.act1 = Act()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm2 = Norm(out_channels)
        self.act2 = Act()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = Identity()

        self.res_weight = self.add_weight(
            name='res_weight', shape=(), dtype=tf.float32,
            trainable=True, initializer=Constant(0.))

    def call(self, x):
        out = self.norm1(x)
        out = self.act1(out)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)
        return out + self.res_weight * shortcut


class ResNet(Model):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, num_classes=10):
        super().__init__()
        num_blocks = (depth - 4) // 6
        self.conv = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, num_blocks, stride=1)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, num_blocks, stride=2)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, num_blocks, stride=2)

        self.norm = Norm(self.stages[3] * k)
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [PreActResBlock(in_channels, out_channels, stride=stride)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, stride=1))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x