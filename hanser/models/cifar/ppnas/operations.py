import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Norm, Pool2d, Conv2d, Act, Identity

OPS = {
    # 'none': lambda C, stride: Zero(stride),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else Pool2d(3, stride, type='avg'),
    'nor_conv_1x1': lambda C, stride: Conv2d(C, C, 1, stride, norm='def', act='def'),
    'nor_conv_3x3': lambda C, stride, dilation=1: Conv2d(C, C, 3, stride, dilation=dilation, norm='def', act='def'),
}