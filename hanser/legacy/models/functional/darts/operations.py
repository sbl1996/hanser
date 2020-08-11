from tensorflow.keras.layers import Multiply, Concatenate
from hanser.models.functional.layers import conv2d, norm, act, pool

OPS = {
    'none': lambda x, C, stride, name: zero(x, stride, name=name),
    'avg_pool_3x3': lambda x, C, stride, name: norm(
        pool(x, 3, stride=stride, type='avg', name=name + "/pool"),
        name=name + "/norm"
    ),
    'max_pool_3x3': lambda x, C, stride, name: norm(
        pool(x, 3, stride=stride, type='max', name=name + "/pool"),
        name=name + "/norm"
    ),
    'skip_connect': lambda x, C, stride, name: x if stride == 1 else factorized_reduce(x, C, name=name),
    'sep_conv_3x3': lambda x, C, stride, name: sep_conv(x, C, 3, stride, name=name),
    'sep_conv_5x5': lambda x, C, stride, name: sep_conv(x, C, 5, stride, name=name),
    'sep_conv_7x7': lambda x, C, stride, name: sep_conv(x, C, 7, stride, name=name),
    'nor_conv_1x1': lambda x, C, stride, name: relu_conv_bn(x, C, 1, stride, name=name),
    'nor_conv_3x3': lambda x, C, stride, name: relu_conv_bn(x, C, 3, stride, name=name),
    'dil_conv_3x3': lambda x, C, stride, name: dil_conv(x, C, 3, stride, 2, name=name),
    'dil_conv_5x5': lambda x, C, stride, name: dil_conv(x, C, 5, stride, 2, name=name),
    'conv_7x1_1x7': lambda x, C, stride, name: conv_7x1_1x7(x, stride, name=name),
}


def conv_7x1_1x7(x, stride, name=None):
    in_channels = x.shape[-1]
    x = act(x, name=name + "/act")
    x = conv2d(x, in_channels, (7, 1), stride=(stride, 1), bias=False, name=name + "/conv_l")
    x = conv2d(x, in_channels, (1, 7), stride=(1, stride), bias=False, name=name + "/conv_r")
    x = norm(x, name=name + "/norm")
    return x


def relu_conv_bn(x, out_channels, kernel_size, stride=1, name=None):
    x = act(x, name=name + "/act")
    x = conv2d(x, out_channels, kernel_size, stride=stride, bias=False, name=name + "/conv")
    x = norm(x, name=name + "/norm")
    return x


def dil_conv(x, out_channels, kernel_size, stride, dilation, name=None):
    in_channels = x.shape[-1]
    x = act(x, name=name + "/act1")
    x = conv2d(x, in_channels, kernel_size, stride=stride, groups=in_channels, dilation=dilation, bias=False,
               name=name + "/depthwise1")
    x = conv2d(x, out_channels, 1, bias=False, name=name + "/pointwise1")
    x = norm(x, name=name + "/norm1")
    return x


def sep_conv(x, out_channels, kernel_size, stride, name=None):
    in_channels = x.shape[-1]

    x = act(x, name=name + "/act1")
    x = conv2d(x, in_channels, kernel_size, stride=stride, groups=in_channels, bias=False, name=name + "/depthwise1")
    x = conv2d(x, in_channels, 1, bias=False, name=name + "/pointwise1")
    x = norm(x, name=name + "/norm1")

    x = act(x, name=name + "/act2")
    x = conv2d(x, in_channels, kernel_size, stride=stride, groups=in_channels, bias=False, name=name + "/depthwise2")
    x = conv2d(x, out_channels, 1, bias=False, name=name + "/pointwise2")
    x = norm(x, name=name + "/norm2")
    return x


def zero(x, stride=1, name=None):
    if stride != 1:
        x = x[:, ::stride, ::stride, :]
    return Multiply(name=name)([x, 0.])


def factorized_reduce(x, out_channels, name=None):
    assert out_channels % 2 == 0
    x = act(x, name=name + "/act")
    x1 = conv2d(x, out_channels // 2, 1, stride=2, bias=False, name=name + "/conv1")
    x2 = conv2d(x[:, 1:, 1:, :], out_channels // 2, 1, stride=2, bias=False, name=name + "/conv2")
    x = Concatenate(name=name + "/concat")([x1, x2])
    x = norm(x, name=name + "/norm")
    return x
