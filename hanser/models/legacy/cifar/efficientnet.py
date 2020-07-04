from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, ReLU, GlobalAvgPool2D, Input, Add, Softmax, Multiply, Dropout

from hanser.models.legacy.layers import bn, conv2d, dense, dwconv2d, swish, sigmoid


def se(x, channels, out_channels):
    s = GlobalAvgPool2D()(x)
    s = Flatten()(s)
    s = dense(s, channels)
    s = ReLU()(s)
    s = dense(s, out_channels)
    s = sigmoid(s)
    x = Multiply()([x, s])
    return x


def mbconv(x, channels, out_channels, kernel_size, stride, expand, se_ratio, drop_rate):
    in_channels = channels // expand
    identity = x
    if expand != 1:
        x = conv2d(x, channels, kernel_size=1)
        x = bn(x)
        x = swish(x)
    x = dwconv2d(x, kernel_size, stride)
    x = bn(x)
    x = swish(x)
    x = se(x, int(in_channels * se_ratio), channels)
    x = conv2d(x, out_channels, kernel_size=1)
    x = bn(x)
    if in_channels == out_channels and stride != 1:
        if drop_rate:
            x = Dropout(drop_rate)(x)
        x = Add()[x, identity]
    return x


def round_channels(channels, multiplier=None, divisor=8, min_depth=None):
    if multiplier is None:
        return channels

    channels *= multiplier
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def efficientnet(input_shape=(32, 32, 3), num_classes=10, width_mult=1.0, drop_connect=0.2):
    setting = [
        # r, k, s, e, o, se,
        [1, 3, 1, 1, 16, 0.25],
        [2, 3, 1, 6, 24, 0.25],
        [2, 5, 1, 6, 40, 0.25],
        [3, 3, 2, 6, 80, 0.25],
        [3, 5, 1, 6, 112, 0.25],
        [4, 5, 2, 6, 192, 0.25],
        [1, 3, 1, 6, 320, 0.25],
    ]

    start_channels = 32
    last_channels = round_channels(1280, width_mult)

    inputs = Input(input_shape)
    x = conv2d(inputs, start_channels, kernel_size=3)
    x = bn(x)
    x = swish(x)

    c = start_channels
    for idx, (r, k, s, e, o, se) in enumerate(setting):
        drop_rate = drop_connect * (float(idx) / len(setting))
        o = round_channels(o, width_mult)
        x = mbconv(x, c * e, o, k, s, e, se, drop_rate)
        c = o
        for _ in range(r - 1):
            x = mbconv(x, c * e, o, k, 1, e, se, drop_rate)

    x = conv2d(x, last_channels, kernel_size=1)
    x = bn(x)
    x = swish(x)

    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = dense(x, num_classes)
    y = Softmax()(x)

    model = Model(inputs=inputs, outputs=y)
    return model






