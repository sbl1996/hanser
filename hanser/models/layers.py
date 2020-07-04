from typing import Union, Tuple, Optional

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.layers import BatchNormalization, Dense, DepthwiseConv2D, Activation, AvgPool2D, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.regularizers import l2

from hanser.models.conv import Conv2D

__all__ = ["DEFAULTS", "Act", "Conv2d", "BN", "Linear"]

DEFAULTS = {
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'fused': True,
    },
    'activation': 'relu',
    'fp16': False,
    'init': {
        'type': 'msra',
        'mode': 'fan_in',
        'uniform': False,
        'std': 0.01,
        'scale': 1.0,
    },
    'seed': 0,
    'no_bias_decay': False,
    'weight_decay': None,
}


def Conv2d(in_channels: int,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           bn: bool = False,
           act: Optional[str] = None,
           zero_init=False,
           name: Optional[str] = None):
    if isinstance(padding, str):
        padding = padding.upper()

    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        distribution = 'uniform' if init_cfg['uniform'] else 'normal'
        kernel_initializer = VarianceScaling(2.0 * init_cfg['scale'], init_cfg['mode'], distribution, DEFAULTS['seed'])
    elif init_cfg['type'] == 'normal':
        kernel_initializer = RandomNormal(0, init_cfg['std'], seed=DEFAULTS['seed'])
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    if bias is False:
        use_bias = False
    elif bias is None and bn:
        use_bias = False
    else:
        use_bias = bias

    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    if not (bn or act):
        conv_name = name
    else:
        conv_name = name + "/conv"
    if in_channels == groups:
        depth_multiplier = out_channels // in_channels
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding,
                               use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                               depthwise_initializer=kernel_initializer, bias_initializer='zeros',
                               kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer, name=conv_name)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding=padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros',
                      kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer, name=conv_name)
    layers = [conv]
    if bn:
        layers.append(BN(out_channels, zero_init=zero_init, name=name + "/bn"))
    if act:
        layers.append(Act(act, name=name + "/act"))
    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(layers, name=name)


def BN(channels, affine=None, zero_init=False, name=None):
    cfg = DEFAULTS['bn']
    if zero_init:
        gamma_initializer = 'zeros'
    else:
        gamma_initializer = 'ones'
    if DEFAULTS['no_bias_decay']:
        gamma_regularizer = None
        beta_regularizer = None
    else:
        gamma_regularizer = get_weight_decay()
        beta_regularizer = get_weight_decay()
    bn = BatchNormalization(
        momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
        gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer, fused=cfg['fused'],
        trainable=affine or cfg['affine'], name=name)
    return bn


def Act(activation='default', name=None):
    if activation == 'default':
        return Act(DEFAULTS['activation'], name)
    return Activation(activation, name=name)


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=False, name=None):

    if isinstance(padding, str):
        padding = padding.upper()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if type == 'avg':
        return AvgPool2D(kernel_size, stride, padding, name=name)
    elif type == 'max':
        return MaxPool2D(kernel_size, stride, padding, name=name)
    else:
        raise ValueError("Unsupported pool type: %s" % type)


def GlobalAvgPool(name=None):
    return GlobalAvgPool2D(name=name)


def get_weight_decay():
    wd = DEFAULTS['weight_decay']
    if wd:
        return l2(wd)


def Linear(in_channels, out_channels, act=None, name=None):
    kernel_initializer = RandomNormal(0, 0.01, seed=DEFAULTS['seed'])
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=get_weight_decay(),
                 bias_regularizer=get_weight_decay(),
                 activity_regularizer=get_weight_decay(),
                 name=name)