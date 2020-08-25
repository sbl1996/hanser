import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from hanser.models.layers import Conv2d, Norm, Act, Pool2d

OPS = {
    'none': lambda C, stride: Zero(stride),
    'avg_pool_3x3': lambda C, stride: Pool2d(3, stride=stride, type='avg'),
    'max_pool_3x3': lambda C, stride: Pool2d(3, stride=stride, type='max'),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride),
    'sep_conv_7x7': lambda C, stride: SepConv(C, C, 7, stride),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 2),
    'conv_7x1_1x7': lambda C, stride: Sequential([
        Act(),
        Conv2d(C, C, (7, 1), stride=(stride, 1), bias=False),
        Conv2d(C, C, (1, 7), stride=(1, stride), bias=False),
        Norm(C),
    ]),
    'nor_conv_1x1': lambda C, stride: ReLUConvBN(C, C, 1, stride),
    'nor_conv_3x3': lambda C, stride: ReLUConvBN(C, C, 3, stride),
    'max_pool_2x2': lambda C, stride: Pool2d(2, stride=stride, type='max'),
}


class ReLUConvBN(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride):
        super().__init__([
            Act(),
            Conv2d(C_in, C_out, kernel_size, stride=stride, bias=False),
            Norm(C_out),
        ])


class DilConv(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, dilation):
        super().__init__([
            Act(),
            Conv2d(C_in, C_in, kernel_size, stride=stride, dilation=dilation, groups=C_in, bias=False),
            Conv2d(C_in, C_out, 1, bias=False),
            Norm(C_out),
        ])


class SepConv(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride):
        super().__init__([
            Act(),
            Conv2d(C_in, C_in, kernel_size, stride=stride, groups=C_in, bias=False),
            Conv2d(C_in, C_in, 1, bias=False),
            Norm(C_in),

            Act(),
            Conv2d(C_in, C_in, kernel_size, 1, groups=C_in, bias=False),
            Conv2d(C_in, C_out, 1, bias=False),
            Norm(C_out),
        ])


class Identity(Layer):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return x


class Zero(Layer):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x):
        if self.stride == 1:
            return x * 0.
        return x[:, ::self.stride, ::self.stride, :] * 0.


class FactorizedReduce(Layer):

    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.act = Act()
        self.conv1 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv2 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.norm = Norm(C_out)

    def call(self, x):
        x = self.act(x)
        x = tf.concat([self.conv1(x), self.conv2(x[:, 1:, 1:, :])], axis=-1)
        x = self.norm(x)
        return x
