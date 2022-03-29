from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal

from hanser.models.layers import Conv2d, Dropout, Linear, GlobalAvgPool


__all__ = ['MobileNetV2', 'mobilenet_v2', 'mobilenet_v2_140', 'mobilenet_v2_050']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(Layer):

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d(in_channels, channels, 1, norm='bn', act='relu6'))
        layers.extend([
            # dw
            Conv2d(channels, channels, 3, stride=stride, groups=channels, norm='bn', act='relu6'),
            # pw-linear
            Conv2d(channels, out_channels, 1, norm='bn'),
        ])
        self.conv = Sequential(layers)

    def call(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Model):
    def __init__(self,
                 num_classes=1000,
                 dropout=0.2,
                 width_mult=1.0,
                 round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            dropout (float): Dropout ratio
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual

        in_channels = 32
        last_channels = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        self.last_channels = _make_divisible(last_channels * max(1.0, width_mult), round_nearest)
        features = [Conv2d(3, in_channels, 3, stride=2, norm='bn', act='relu6')]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channels, output_channel, stride, expand_ratio=t))
                in_channels = output_channel
        features.append(Conv2d(in_channels, self.last_channels, 1, norm='bn', act='relu6'))
        self.features = Sequential(features)

        # building classifier

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(self.last_channels, num_classes,
                         kernel_init=RandomNormal(stddev=0.01), bias_init='zeros')

    def call(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


def mobilenet_v2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model

def mobilenet_v2_140(**kwargs):
    model = MobileNetV2(width_mult=1.4, **kwargs)
    return model

def mobilenet_v2_050(**kwargs):
    model = MobileNetV2(width_mult=0.5, **kwargs)
    return model