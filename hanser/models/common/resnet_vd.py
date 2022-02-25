from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Norm
from hanser.models.common.modules import get_shortcut_vd

class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride, zero_init_residual=False, dilation=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            norm='def', gamma_init='zeros' if zero_init_residual else 'ones')

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, zero_init_residual=False, dilation=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            dilation=dilation, norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros' if zero_init_residual else 'ones')

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

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