from tensorflow.keras.layers import Layer

from hanser.models.modules import DropPath
from hanser.models.layers import Conv2d, Act, Identity, Norm, NormAct
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.attention import SELayer


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride,
                 start_block=False, end_block=False, exclude_bn0=False,
                 drop_path=0, se_reduction=4, se_mode=0, se_last=False):
        super().__init__()
        self.se_last = se_last

        out_channels = channels * self.expansion

        if not start_block:
            if exclude_bn0:
                self.act0 = Act()
            else:
                self.norm_act0 = NormAct(in_channels)

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def', avd=True, avd_first=False)

        if not self.se_last:
            self.se = SELayer(channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if self.se_last:
            self.se = SELayer(out_channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

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
        if not self.se_last:
            x = self.se(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        if self.se_last:
            x = self.se(x)

        x = self.drop_path(x)
        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x