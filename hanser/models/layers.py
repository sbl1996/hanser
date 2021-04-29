import math
from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping

from cerberus import Validator

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Initializer
from tensorflow.keras.layers import Dense, Activation, Layer, Conv2D, ZeroPadding2D, LeakyReLU, \
    DepthwiseConv2D, MaxPooling2D as KerasMaxPool2D, AveragePooling2D as KerasAvgPool2D
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish

from hanser.models.pooling import MaxPooling2D as MaxPool2D, AveragePooling2D as AvgPool2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization

__all__ = ["set_default", "set_defaults", "Act", "Conv2d", "Norm", "Linear", "GlobalAvgPool", "Pool2d", "Identity"]

DEFAULTS = {
    'naive_padding': False,
    'conv': {
        'depthwise': {
            'use_group': False,
            'horch': False,
        },
        'group': {
            'smart_naive': False,
            'max_naive_groups': 8,
        },
        'init': {
            'type': 'msra',
            'mode': 'fan_out',
            'distribution': 'untruncated_normal',
            'fix': True,
        },
    },
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
        'fused': True,
        'sync': False,
        'eval': False,
    },
    'gn': {
        'groups': 32,
        'channels_per_group': 16,
        'eps': 1e-5,
        'affine': True,
    },
    'activation': 'relu',
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
}

_defaults_schema = {
    'naive_padding': {'type': 'boolean'},
    'conv': {
        'depthwise': {
            'use_group': {'type': 'boolean'},
            'horch': {'type': 'boolean'},
        },
        'group': {
            'smart_naive': {'type': 'boolean'},
            'max_naive_groups': {'type': 'integer'},
        },
        'init': {
            'type': {'type': 'string', 'allowed': ['msra', 'normal']},
            'mode': {'type': 'string', 'allowed': ['fan_in', 'fan_out']},
            'distribution': {'type': 'string', 'allowed': ['uniform', 'truncated_normal', 'untruncated_normal']},
            'fix': {'type': 'boolean'},
        },

    },
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
        'fused': {'type': 'boolean'},
        'sync': {'type': 'boolean'},
        'eval': {'type': 'boolean'},
    },
    'gn': {
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'groups': {'type': 'integer'},
        'channels_per_group': {'type': 'integer'},
    },
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn', 'gn', 'none']},
    'seed': {'type': 'integer'},
}


def set_defaults(kvs: Mapping):
    def _set_defaults(kvs, prefix):
        for k, v in kvs.items():
            if isinstance(v, dict):
                _set_defaults(v, prefix + (k,))
            else:
                set_default(prefix + (k,), v)

    return _set_defaults(kvs, ())


def set_default(keys: Union[str, Sequence[str]], value):
    def loop(d, keys, schema):
        k = keys[0]
        if k not in d:
            match = get_close_matches(k, d.keys())
            if match:
                raise KeyError("No such key `%s`, maybe you mean `%s`" % (k, match[0]))
            else:
                raise KeyError("No key `%s` in %s" % (k, d))
        if len(keys) == 1:
            v = Validator({k: schema[k]})
            if not v.validate({k: value}):
                raise ValueError(v.errors)
            d[k] = value
        else:
            loop(d[k], keys[1:], schema[k])

    if isinstance(keys, str):
        keys = [keys]

    if len(keys) == 1 and keys[0] == 'activation':
        DEFAULTS['activation'] = value
        return
    loop(DEFAULTS, keys, _defaults_schema)


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = ((ph, ph), (pw, pw))
    return padding


def flip_mode(m):
    if m == 'fan_in':
        return 'fan_out'
    else:
        return 'fan_in'


def Conv2d(in_channels: int,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[str] = None,
           kernel_init: Optional[Initializer] = None,
           bias_init: Optional[Initializer] = None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, tuple) and isinstance(padding[0], int):
        assert len(padding) == 2
        ph, pw = padding
        padding = ((ph, ph), (pw, pw))

    conv_cfg = DEFAULTS['conv']
    init_cfg = conv_cfg['init']
    naive_padding = DEFAULTS['naive_padding'] or padding == 'SAME'

    if naive_padding:
        conv_padding = padding
    else:
        conv_padding = 'valid'

    if isinstance(padding, str):
        assert padding in ['same', 'SAME']
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    # Init
    if kernel_init:
        kernel_initializer = kernel_init
    elif init_cfg['type'] == 'msra':
        mode = init_cfg['mode']
        distribution = init_cfg['distribution']
        if in_channels == groups and not conv_cfg['depthwise']['use_group'] and init_cfg['fix']:
            mode = flip_mode(mode)
        if 'uniform' in distribution:
            kernel_initializer = VarianceScaling(1.0 / 3, mode, distribution)
        else:
            kernel_initializer = VarianceScaling(2.0, mode, distribution)
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    if bias_init:
        bias_initializer = bias_init
    else:
        bound = math.sqrt(1 / (kernel_size[0] * kernel_size[1] * (in_channels // groups)))
        bias_initializer = RandomUniform(-bound, bound)

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    if in_channels == groups:
        if conv_cfg['depthwise']['use_group']:
            conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                          padding=conv_padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        else:
            depth_multiplier = out_channels // in_channels
            if conv_cfg['depthwise']['horch']:
                from hanser.models.conv import DepthwiseConv2D as HorchDepthwiseConv2D
                depth_conv = HorchDepthwiseConv2D
            else:
                depth_conv = DepthwiseConv2D
            conv = depth_conv(kernel_size=kernel_size, strides=stride, padding=conv_padding,
                              use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                              depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
    elif conv_cfg['group']['smart_naive'] and 1 < groups <= conv_cfg['group']['max_naive_groups']:
        conv = NaiveGroupConv2D(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=conv_padding, groups=groups)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding=conv_padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    if not naive_padding and padding != ((0, 0), (0, 0)):
        conv = Sequential([
            ZeroPadding2D(padding),
            conv,
        ])

    layers = [conv]
    if norm:
        layers.append(Norm(out_channels, norm))
    if act:
        layers.append(Act(act))

    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(layers)


def get_groups(channels, ref=32):
    if channels == 1:
        return 1
    xs = filter(lambda x: channels % x == 0, range(2, channels + 1))
    c = min(xs, key=lambda x: abs(x - ref))
    if c < 8:
        c = max(c, channels // c)
    return channels // c


def Norm(channels, type='default', affine=None, track_running_stats=None, gamma_init='ones'):
    if type in ['default', 'def']:
        type = DEFAULTS['norm']
    if type == 'bn':
        cfg = DEFAULTS['bn']
        if affine is None:
            affine = cfg['affine']
        if track_running_stats is None:
            track_running_stats = cfg['track_running_stats']
        if cfg['sync']:
            bn = SyncBatchNormalization(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init, track_running_stats=track_running_stats,
                eval_mode=cfg['eval'])
        else:
            bn = BatchNormalization(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init, fused=cfg['fused'], track_running_stats=track_running_stats,
                eval_mode=cfg['eval'])
        return bn
    elif type == 'gn':
        cfg = DEFAULTS['gn']
        if affine is None:
            affine = cfg['affine']
        if not cfg['groups']:
            groups = get_groups(channels, cfg['channels_per_group'])
        else:
            groups = cfg['groups']
        gn = tfa.layers.GroupNormalization(
            groups=groups, epsilon=cfg['eps'], center=affine, scale=affine)
        return gn
    elif type == 'none':
        return Identity()
    else:
        raise ValueError("Unsupported normalization type: %s" % type)

def Act(type='default', **kwargs):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'], **kwargs)
    if type == 'mish':
        return Mish()
    elif type == 'leaky_relu':
        if 'alpha' not in kwargs:
            kwargs = {**kwargs, 'alpha': DEFAULTS['leaky_relu']['alpha']}
        return LeakyReLU(**kwargs)
    else:
        return Activation(type, **kwargs)


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=True):
    assert padding == 0 or padding == 'same'
    if ceil_mode:
        assert padding == 'same'
    else:
        assert padding == 0

    if padding == 0:
        padding = 'valid'

    if type == 'avg':
        pool = KerasAvgPool2D if DEFAULTS['naive_padding'] else AvgPool2D
    elif type == 'max':
        pool = KerasMaxPool2D if DEFAULTS['naive_padding'] else MaxPool2D
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    return pool(kernel_size, stride, padding)


class GlobalAvgPool(Layer):
    """Abstract class for different global pooling 2D layers.
  """

    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    # noinspection PyMethodOverriding
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=self.keep_dim)

    def get_config(self):
        config = {'keep_dim': self.keep_dim}
        base_config = super().get_config()
        return {**base_config, **config}


class Identity(Layer):
    """Abstract class for different global pooling 2D layers.
  """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)


def Linear(in_channels, out_channels, act=None, kernel_init=None, bias_init=None):
    kernel_initializer = kernel_init or VarianceScaling(1.0 / 3, 'fan_in', 'uniform')
    bound = math.sqrt(1 / in_channels)
    bias_initializer = bias_init or RandomUniform(-bound, bound)
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer)


class Mish(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        return mish(x)

    def get_config(self):
        base_config = super().get_config()
        return base_config


class NaiveGroupConv2D(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        D_out = out_channels // groups
        self.convs = [
            Conv2D(D_out, kernel_size=kernel_size, strides=stride, padding=padding)
            for _ in range(groups)
        ]

    def call(self, x):
        # C = self.in_channels // self.groups
        # x = tf.concat([
        #     conv(x[:, :, :, i * C: (i + 1) * C])
        #     for i, conv in enumerate(self.convs)
        # ], axis=-1)
        xs = tf.split(x, self.groups, axis=-1)
        xs = [
            conv(x) for conv, x in zip(self.convs, xs)
        ]
        x = tf.concat(xs, axis=-1)
        return x