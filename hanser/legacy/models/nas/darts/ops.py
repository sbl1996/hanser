from tensorflow.keras.layers import MaxPool2D, AvgPool2D

from hanser.models.legacy.layers import bn, conv2d, dwconv2d, relu, cat


def sep_conv(x, channels, kernel_size, stride):
    x = relu(x)
    x = dwconv2d(x, kernel_size, stride=stride)
    x = conv2d(x, channels, 1)
    x = bn(x, affine=False)
    x = relu(x)
    x = dwconv2d(x, kernel_size)
    x = conv2d(x, channels, 1)
    x = bn(x, affine=False)
    return x


def dil_conv(x, channels, kernel_size, stride, dilation):
    assert stride == 1
    x = relu(x)
    x = dwconv2d(x, kernel_size, stride=1, dilation=dilation)
    x = conv2d(x, channels, 1)
    x = bn(x, affine=False)
    return x


def factorized_reduce(x, channels):
    assert channels % 2 == 0
    x = relu(x)
    x = cat([conv2d(x, channels // 2, 1, 2), conv2d(x[:, 1:, 1:, :], channels // 2, 1, 2)])
    x = bn(x, affine=False)
    return x


def sep_conv_3x3(x, channels, stride):
    return sep_conv(x, channels, 3, stride)


def zero(x, stride):
    if stride == 2:
        x = x[:, ::2, ::2, :]
    return x * 0


def sep_conv_5x5(x, channels, stride):
    return sep_conv(x, channels, 3, stride)


def skip_connect(x, channels, stride):
    return x if stride == 1 else factorized_reduce(x, channels)


def dil_conv_3x3(x, channels):
    return dil_conv(x, channels, 3, 1, 2)


def avg_pool_3x3(x, stride):
    return AvgPool2D(3, stride, padding='same')(x)


def max_pool_3x3(x, stride):
    return MaxPool2D(3, stride, padding='same')(x)


NORMAL_OPS = {
    'none': lambda x, c: zero(x, 1),
    'skip_connect': lambda x, c: skip_connect(x, c, 1),
    'sep_conv_3x3': lambda x, c: sep_conv(x, c, 3, 1),
    'sep_conv_5x5': lambda x, c: sep_conv(x, c, 5, 1),
    'sep_conv_7x7': lambda x, c: sep_conv(x, c, 7, 1),
    'avg_pool_3x3': lambda x, c: avg_pool_3x3(x, 1),
    'max_pool_3x3': lambda x, c: max_pool_3x3(x, 1),
}

REDUCTION_OPS = {
    'none': lambda x, c, s: zero(x, s),
    'skip_connect': lambda x, c, s: skip_connect(x, c, s),
    'avg_pool_3x3': lambda x, c, s: avg_pool_3x3(x, s),
    'max_pool_3x3': lambda x, c, s: max_pool_3x3(x, s),
}