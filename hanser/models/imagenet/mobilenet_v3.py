from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal

from hanser.models.attention import SELayer
from hanser.models.layers import Conv2d, Identity, GlobalAvgPool, Linear, Dropout


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(Layer):
    def __init__(self, in_channels, channels, out_channels, kernel_size, stride, act='relu', with_se=True):
        super().__init__()
        self.with_se = with_se
        if in_channels != channels:
            self.expand = Conv2d(in_channels, channels, kernel_size=1,
                                 norm='bn', act=act)
        else:
            self.expand = Identity()

        self.dwconv = Conv2d(channels, channels, kernel_size, stride, groups=channels,
                             norm='bn', act=act)

        if self.with_se:
            self.se = SELayer(channels, reduction=4, act='relu', gating_fn='hsigmoid',
                              min_se_channels=8, divisible=8)

        self.project = Conv2d(channels, out_channels, kernel_size=1,
                              norm='bn')
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def call(self, x):
        identity = x
        x = self.expand(x)
        x = self.dwconv(x)
        if self.with_se:
            x = self.se(x)
        x = self.project(x)
        if self.use_res_connect:
            x += identity
        return x


class MobileNetV3(Model):
    def __init__(self, inverted_residual_setting, last_channels, num_classes=1000, width_mult=1.0, dropout=0.2):
        super().__init__()
        block = InvertedResidual
        in_channels = 16

        last_channels = _make_divisible(last_channels * width_mult) if width_mult > 1.0 else last_channels

        # building first layer
        features = [Conv2d(3, in_channels, kernel_size=3, stride=1,
                           norm='bn', act='hswish')]
        # building inverted residual blocks
        for k, exp, c, se, nl, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult)
            exp_channels = _make_divisible(exp * width_mult)
            features.append(block(
                in_channels, exp_channels, out_channels, k, s, nl, se))
            in_channels = out_channels
        # building last several layers
        features.extend([
            Conv2d(in_channels, exp_channels, kernel_size=1,
                   norm='bn', act='hswish'),
        ])
        in_channels = exp_channels
        self.features = Sequential(features)

        self.avgpool = GlobalAvgPool()
        self.last_fc = Linear(in_channels, last_channels, act='hswish')
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(last_channels, num_classes,
                         kernel_init=RandomNormal(stddev=0.01), bias_init='zeros')

    def call(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.last_fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


def mobilenet_v3_large(mult=1.0, **kwargs):
    inverted_residual_setting = [
        # k, e, o,  se,     nl,  s,
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', 2],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', 2],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hswish', 2],
        [3, 200, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 480, 112, True, 'hswish', 1],
        [3, 672, 112, True, 'hswish', 1],
        [5, 672, 160, True, 'hswish', 2],
        [5, 960, 160, True, 'hswish', 1],
        [5, 960, 160, True, 'hswish', 1],
    ]
    last_channels = 1280
    return MobileNetV3(inverted_residual_setting, last_channels, width_mult=mult, **kwargs)


def mobilenet_v3_small(mult=1.0, **kwargs):
    inverted_residual_setting = [
        # k, e, o,  se,     nl,  s,
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hswish', 2],
        [5, 240, 40, True, 'hswish', 1],
        [5, 240, 40, True, 'hswish', 1],
        [5, 120, 48, True, 'hswish', 1],
        [5, 144, 48, True, 'hswish', 1],
        [5, 288, 96, True, 'hswish', 1],
        [5, 576, 96, True, 'hswish', 1],
        [5, 576, 96, True, 'hswish', 1],
    ]
    last_channels = 1024
    return MobileNetV3(inverted_residual_setting, last_channels, width_mult=mult, **kwargs)
