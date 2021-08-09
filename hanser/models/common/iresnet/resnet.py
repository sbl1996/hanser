from tensorflow.keras.layers import Layer, Dropout

from hanser.models.modules import DropPath
from hanser.models.layers import Conv2d, Act, Identity, Norm
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.attention import SELayer


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 pool_type='max', dropout=0, drop_path=0, anti_alias=False,
                 se_reduction=None):
        super().__init__()

        out_channels = channels * self.expansion
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)

        if not start_block:
            self.act0 = Act()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='def', act='def', anti_alias=anti_alias)

        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if start_block:
            self.bn2 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.bn2 = Norm(out_channels)
            self.act2 = Act()

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride,
                                        pool_type=pool_type)

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
        x = self.dropout(x)

        x = self.conv2(x)
        if self.start_block:
            x = self.bn2(x)
        x = self.drop_path(x)
        x = x + identity
        if self.end_block:
            x = self.bn2(x)
            x = self.act2(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 pool_type='max', drop_path=0, anti_alias=False,
                 se_reduction=None):
        super().__init__()

        out_channels = channels * self.expansion

        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)
        if not start_block:
            self.act0 = Act()

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', anti_alias=anti_alias)

        self.se = SELayer(channels, se_channels=out_channels // se_reduction) \
            if se_reduction is not None else None

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.bn3 = Norm(out_channels)
            self.act3 = Act()

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride,
                                        pool_type=pool_type)

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
        if self.se is not None:
            x = self.se(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        x = self.drop_path(x)
        x = x + identity

        if self.end_block:
            x = self.bn3(x)
            x = self.act3(x)
        return x