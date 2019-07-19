from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Flatten, ReLU, GlobalAvgPool2D, Input, AvgPool2D, Add, Softmax

from hanser.model.layers import PadChannel, bn, conv2d, dense


def shortcut(x, out_channels, stride=1):
    if stride == 2:
        x = AvgPool2D()(x)
    x = PadChannel(out_channels)(x)
    return x


def basicblock(x, out_channels, stride=1):
    identity = x
    x = bn()(x)
    x = conv2d(out_channels, kernel_size=3)(x)
    x = bn()(x)
    x = ReLU()(x)
    x = conv2d(out_channels, kernel_size=3, stride=stride)(x)
    x = bn(gamma='zeros')(x)
    identity = shortcut(identity, out_channels, stride=stride)
    return Add()([x, identity])


def bottleneck(x, channels, stride=1):
    identity = x
    out_channels = channels * 4
    x = bn()(x)
    x = conv2d(channels, kernel_size=1)(x)
    x = bn()(x)
    x = ReLU()(x)
    x = conv2d(channels, kernel_size=3, stride=stride)(x)
    x = bn()(x)
    x = ReLU()(x)
    x = conv2d(out_channels, kernel_size=1)(x)
    x = bn(gamma='zeros')(x)
    identity = shortcut(identity, out_channels, stride=stride)
    return Add()([x, identity])


def rd(c):
    return int(round(c, 2))


def pyramidnet(input_shape=(32, 32, 3), num_classes=10, start_channels=16, widening_fractor=84, num_layers=(18, 18, 18), block='basic'):
    assert len(num_layers) == 3
    assert block in ["basic", "bottleneck"]
    if block == "basic":
        block = basicblock
    elif block == "bottleneck":
        block = bottleneck

    add_channels = widening_fractor / sum(num_layers)
    channels = start_channels

    inputs = Input(input_shape)
    x = conv2d(start_channels, kernel_size=3)(inputs)
    x = bn()(x)

    strides = [1, 2, 2]

    for s, l in zip(strides, num_layers):
        for i in range(l):
            channels += add_channels
            x = block(x, rd(channels), stride=s if i == 0 else 1)

    x = bn()(x)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = dense(num_classes)(x)
    y = Softmax()(x)

    model = Model(inputs=inputs, outputs=y)
    return model
