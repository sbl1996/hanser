import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

from hanser.models.layers import Conv2d, Act, Identity, Norm, NormAct
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.modules import DropPath


from hanser.models.nas.simple.iresnet import _IResNet


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 drop_path=0):
        super().__init__()

        out_channels = channels * self.expansion

        if not start_block:
            if exclude_bn0:
                self.act0 = Act()
            else:
                self.norm_act0 = NormAct(in_channels)

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', anti_alias=True)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()
        self.gate = self.add_weight(
            name='gate', shape=(), trainable=False, initializer=Constant(1.))

        if end_block:
            self.norm_act3 = NormAct(out_channels)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x):
        identity = self.shortcut(x)

        if not self.start_block:
            if self.exclude_bn0:
                x = self.act0(x)
            else:
                x = self.norm_act0(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        x = self.drop_path(x)
        x = x * tf.cast(self.gate, x.dtype)
        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x


class ResNetCIFAR(_IResNet):

    def __init__(self, depth, drop_layer, num_classes=100, channels=(16, 16, 32, 64),
                 drop_path=0.5, **kwargs):
        depths = [(depth - 2) // 9] * 3
        depths = [(d - drop_layer, d) for d in depths]

        stem_channels, *channels = channels
        stem = Conv2d(3, stem_channels, kernel_size=3, norm='def', act='def')
        stem.out_channels = stem_channels
        super().__init__(stem, Bottleneck, depths, num_classes, channels,
                         strides=(1, 2, 2), drop_path=drop_path, **kwargs)


def resnet(depth=110, drop_layer=6, drop_path=0.5, **kwargs):
    return ResNetCIFAR(depth, drop_layer, drop_path=drop_path, **kwargs)