import math
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act, Identity
from hanser.models.common.modules import get_shortcut


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, cardinality, base_width, zero_init_residual=False):
        super().__init__()
        out_channels = channels * self.expansion

        D = math.floor(channels * (base_width / 64))
        C = cardinality

        self.conv1 = Conv2d(in_channels, D * C, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(D * C, D * C, kernel_size=3, stride=stride, groups=cardinality,
                            norm='def', act='def')
        self.conv3 = Conv2d(D * C, out_channels, kernel_size=1,
                            norm='def', gamma_init='zeros' if zero_init_residual else 'ones')

        self.shortcut = get_shortcut(in_channels, out_channels, stride)

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.act(x)
        return x