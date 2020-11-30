from typing import Union, Tuple, Optional, Callable

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.layers import BatchNormalization, Dense, DepthwiseConv2D, \
    Activation, AvgPool2D, MaxPool2D, Layer, Conv2D
from tensorflow.keras.regularizers import l2
from tensorflow_addons.activations import mish

__all__ = ["DEFAULTS", "act", "norm", "conv2d", "dense", "pool"]

DEFAULTS = {
    'norm': 'bn',
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'fused': True,
        'sync': False,
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


class Mish(Layer):

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, x, training=None):
        return mish(x)


def conv2d(x,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[Union[str, Callable]] = None,
           zero_init=False,
           name: Optional[str] = None):
    in_channels = x.shape[-1]

    if isinstance(padding, str):
        padding = padding.upper()

    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        distribution = 'uniform' if init_cfg['uniform'] else 'normal'
        kernel_initializer = VarianceScaling(2.0 * init_cfg['scale'], init_cfg['mode'], distribution)
    elif init_cfg['type'] == 'normal':
        kernel_initializer = RandomNormal(0, init_cfg['std'])
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    if bias is False:
        use_bias = False
    elif bias is None and norm:
        use_bias = False
    else:
        use_bias = bias or True

    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    if not (norm or act):
        conv_name = name
    else:
        conv_name = name + "/conv"
    activation = act if (act is not None and norm is None) else None
    if in_channels == groups:
        depth_multiplier = out_channels // in_channels
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                               dilation_rate=dilation, activation=activation, depth_multiplier=depth_multiplier,
                               depthwise_initializer=kernel_initializer, bias_initializer='zeros',
                               kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer,
                               name=conv_name)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding=padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros', activation=activation,
                      kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer, name=conv_name)
    x = conv(x)
    if norm:
        x = norm_(x, norm, zero_init=zero_init, name=name + "/bn")
        if act:
            x = act_(x, act, name=name + "/act")
    return x


def norm_(x, type='default', affine=None, zero_init=False, name=None):
    if type == 'default':
        type = DEFAULTS['norm']
    assert type == 'bn'
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
    if cfg['sync']:
        layer = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer,
            trainable=affine or cfg['affine'], name=name)
    else:
        layer = BatchNormalization(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer, fused=cfg['fused'],
            trainable=affine or cfg['affine'], name=name)
    return layer(x)


norm = norm_


def act_(x, type='default', name=None):
    if type == 'default':
        return act_(x, DEFAULTS['activation'], name)
    try:
        layer = Activation(type, name=name)(x)
    except ValueError:
        if type == 'mish':
            layer = Mish(name=name)(x)
        else:
            raise ValueError("Unknown activation function:%s" % type)
    return layer


act = act_


# noinspection PyUnusedLocal
def pool(x, kernel_size, stride, padding='same', type='avg', ceil_mode=False, name=None):
    if isinstance(padding, str):
        padding = padding.upper()

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if type == 'avg':
        return AvgPool2D(kernel_size, stride, padding, name=name)(x)
    elif type == 'max':
        return MaxPool2D(kernel_size, stride, padding, name=name)(x)
    else:
        raise ValueError("Unsupported pool type: %s" % type)


def get_weight_decay():
    wd = DEFAULTS['weight_decay']
    if wd:
        return l2(wd)


def dense(x, out_channels, act=None, name=None):
    kernel_initializer = RandomNormal(0, 0.01)
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=get_weight_decay(),
                 bias_regularizer=get_weight_decay(),
                 name=name)(x)
