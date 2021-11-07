import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

from hanser.train.callbacks import Callback
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.layers import Conv2d, GlobalAvgPool, Linear, Act, Identity, Norm
from hanser.models.modules import DropPath


class DropPathRateSchedule(Callback):

    def __init__(self, drop_path):
        super().__init__()
        self.drop_path = drop_path

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        rate = (epoch + 1) / epochs * self.drop_path
        for l in self.learner.model.submodules:
            if isinstance(l, DropPath):
                l.rate.assign(rate)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 drop_path=0):
        super().__init__()

        out_channels = channels * self.expansion

        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)
        if not start_block:
            self.act0 = Act()

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def')

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()
        self.gate = self.add_weight(
            name='gate', shape=(), trainable=False, initializer=Constant(1.))

        if end_block:
            self.bn3 = Norm(out_channels)
            self.act3 = Act()

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x):
        identity = self.shortcut(x)

        if not self.start_block:
            if not self.exclude_bn0:
                x = self.bn0(x)
            x = self.act0(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        x = self.drop_path(x)
        x = x * tf.cast(self.gate, x.dtype)
        x = x + identity

        if self.end_block:
            x = self.bn3(x)
            x = self.act3(x)
        return x


class ResNet(Model):

    def __init__(self, depth, drop_path=0, num_classes=100, stages=(16, 16, 32, 64)):
        super().__init__()
        self.stages = stages
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            drop_path=drop_path)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            drop_path=drop_path)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            drop_path=drop_path)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, start_block=True,
                        **{**kwargs, "drop_path": 0})]
        self.in_channels = channels * block.expansion

        for i in range(1, blocks - 1):
            i_kwargs = {**kwargs}
            if "drop_path" in i_kwargs and i < 10:
                i_kwargs['drop_path'] = 0
            layers.append(block(self.in_channels, channels, stride=1,
                                exclude_bn0=i == 1, **i_kwargs))

        layers.append(block(self.in_channels, channels, stride=1,
                            end_block=True, **kwargs))

        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def resnet110(**kwargs):
    return ResNet(depth=110, **kwargs)