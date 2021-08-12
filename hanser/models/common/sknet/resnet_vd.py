from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.common.sknet.layers import SKConv


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, reduction=2):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = SKConv(channels, channels, kernel_size=3, stride=stride,
                            reduction=reduction, norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1,
                            norm='def')

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x
