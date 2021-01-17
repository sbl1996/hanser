import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Norm, Pool2d, Conv2d, Act, Identity

OPS = {
    # 'none': lambda C, stride: Zero(stride),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else Pool2d(3, stride, type='avg'),
    'nor_conv_1x1': lambda C, stride: Conv2d(C, C, 1, stride, norm='def', act='def'),
    'nor_conv_3x3': lambda C, stride: Conv2d(C, C, 3, stride, norm='def', act='def'),
}


# class DilConv(Layer):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, dilation):
#         super().__init__()
#         self.op = Sequential([
#             Act('relu'),
#             Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=C_in, bias=False),
#             Conv2d(C_in, C_out, kernel_size=1, bias=False),
#             Norm(C_out),
#         ])
#
#     def forward(self, x):
#         return self.op(x)
#
#
# class SepConv(Layer):
#
#     def __init__(self, C_in, C_out, kernel_size, stride):
#         super().__init__()
#         self.op = Sequential([
#             Act('relu'),
#             Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, groups=C_in, bias=False),
#             Conv2d(C_in, C_in, kernel_size=1, bias=False),
#             Norm(C_in),
#             Act('relu'),
#             Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, groups=C_in, bias=False),
#             Conv2d(C_in, C_out, kernel_size=1, bias=False),
#             Norm(C_out),
#         ])
#
#     def forward(self, x):
#         return self.op(x)
#
#
# class Zero(Layer):
#
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x * 0.
#         return x[:, :, ::self.stride, ::self.stride] * 0.
#
#
# class FactorizedReduce(Layer):
#
#     def __init__(self, C_in, C_out):
#         super().__init__()
#         assert C_out % 2 == 0
#         self.act = Act('relu')
#         self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
#         self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
#         self.bn = Norm(C_out)
#
#     def forward(self, x):
#         x = self.act(x)
#         out = tf.concat([self.conv_1(x), self.conv_2(x[:, 1:, 1:, :])], axis=-1)
#         out = self.bn(out)
#         return out
