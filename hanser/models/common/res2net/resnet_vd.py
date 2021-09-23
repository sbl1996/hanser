import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Act, Pool2d, Norm
from hanser.models.common.res2net.layers import Res2Conv


class Bottle2neck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, dilation=1, base_width=26, scale=4,
                 zero_init_residual=False):
        super().__init__()
        out_channels = channels * self.expansion
        start_block = stride != 1 or in_channels != out_channels
        width = math.floor(channels * (base_width / 64)) * scale
        self.conv1 = Conv2d(in_channels, width, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Res2Conv(width, width, kernel_size=3, stride=stride, dilation=dilation,
                              scale=scale, groups=1, start_block=start_block, norm='def', act='def')
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros' if zero_init_residual else 'ones')

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.act(x)
        return x