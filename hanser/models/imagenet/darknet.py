import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d
from hanser.models.modules import GlobalAvgPool, Dropout


class BasicBlock(Layer):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1, norm='def', act='def')
        self.conv2 = Conv2d(channels, in_channels, kernel_size=3, norm='def', act='def')

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x


def make_stage(block, in_channels, channels, num_layers, **kwargs):
    layers = [
        Conv2d(in_channels, channels, kernel_size=3, stride=2, norm='def', act='def')]
    for i in range(num_layers):
        layers.append(block(channels, channels // 2, **kwargs))
    return Sequential(layers)


class DarkNet(Model):

    def __init__(self, layers=(1, 2, 8, 8, 4), channels=(64, 128, 256, 512, 1024), num_classes=1000, dropout=0.0):
        super(DarkNet, self).__init__()

        self.stem = Conv2d(3, 32, kernel_size=3, norm='def', act='def')
        in_channels = 32

        for i, (n, c) in enumerate(zip(layers, channels)):
            stage = make_stage(BasicBlock, in_channels, c, n)
            setattr(self, 'stage{}'.format(i + 1), stage)
            in_channels = c

        self.avgpool = GlobalAvgPool(keep_dim=True)
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Conv2d(in_channels, num_classes, kernel_size=1)

    def call(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = tf.squeeze(x, axis=(1, 2))
        return x


def darknet53(**kwargs):
    return DarkNet(layers=(1, 2, 8, 8, 4), channels=(64, 128, 256, 512, 1024), **kwargs)