from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, concatenate
from hanser.models.layers import Conv2d, Norm, Act, Pool2d

OPS = {
    'none': lambda C, stride, name: Zero(stride, name=name),
    'avg_pool_3x3': lambda C, stride, name: Pool2d(3, stride=stride, type='avg', name=name),
    'max_pool_3x3': lambda C, stride, name: Pool2d(3, stride=stride, type='max', name=name),
    'skip_connect': lambda C, stride, name: Identity(name=name) if stride == 1 else FactorizedReduce(C, C, name=name),
    'sep_conv_3x3': lambda C, stride, name: SepConv(C, C, 3, stride, name=name),
    'sep_conv_5x5': lambda C, stride, name: SepConv(C, C, 5, stride, name=name),
    'sep_conv_7x7': lambda C, stride, name: SepConv(C, C, 7, stride, name=name),
    'dil_conv_3x3': lambda C, stride, name: DilConv(C, C, 3, stride, 2, name=name),
    'dil_conv_5x5': lambda C, stride, name: DilConv(C, C, 5, stride, 2, name=name),
    'conv_7x1_1x7': lambda C, stride, name: Sequential([
        Act(name='act'),
        Conv2d(C, C, (7, 1), stride=(stride, 1), bias=False, name='conv_left'),
        Conv2d(C, C, (1, 7), stride=(1, stride), bias=False, name='conv_right'),
        Norm(C, name='norm'),
    ], name=name),
}


class ReLUConvBN(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, name):
        super().__init__([
            Act(name='act'),
            Conv2d(C_in, C_out, kernel_size, stride=stride, bias=False, name='conv'),
            Norm(C_out, name='norm'),
        ], name=name)


class DilConv(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, dilation, name):
        super().__init__([
            Act(name='act'),
            Conv2d(C_in, C_in, kernel_size, stride=stride, dilation=dilation, groups=C_in,
                   bias=False, name='depthwise'),
            Conv2d(C_in, C_out, 1, bias=False, name='pointwise'),
            Norm(C_out, name='norm'),
        ], name=name)


class SepConv(Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, name):
        super().__init__([
            Act(name='act1'),
            Conv2d(C_in, C_in, kernel_size, stride=stride, groups=C_in, bias=False, name='depthwise1'),
            Conv2d(C_in, C_in, 1, bias=False, name='pointwise1'),
            Norm(C_in, name='norm1'),

            Act(name='act2'),
            Conv2d(C_in, C_in, kernel_size, 1, groups=C_in, bias=False, name='depthwise2'),
            Conv2d(C_in, C_out, 1, bias=False, name='pointwise2'),
            Norm(C_in, name='norm2'),
        ], name=name)


class Identity(Layer):

    def __init__(self, name):
        super().__init__(name=name)

    def call(self, x):
        return x


class Zero(Layer):

    def __init__(self, stride, name):
        super().__init__(name=name)
        self.stride = stride

    def call(self, x):
        if self.stride == 1:
            return x * 0.
        return x[:, ::self.stride, ::self.stride, :] * 0.


class FactorizedReduce(Layer):

    def __init__(self, C_in, C_out, name):
        super().__init__(name=name)
        assert C_out % 2 == 0
        self.act = Act(name='act')
        self.conv1 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False, name='conv1')
        self.conv2 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False, name='conv2')
        self.norm = Norm(C_out, name='norm')

    def call(self, x):
        x = self.act(x)
        x = concatenate([self.conv1(x), self.conv2(x[:, 1:, 1:, :])], axis=-1)
        x = self.norm(x)
        return x
