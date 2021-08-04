import math
from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping, Dict, Any
from toolz import curry

from cerberus import Validator

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Initializer, Constant
from tensorflow.keras.layers import Dense, Activation, Layer, Conv2D, ZeroPadding2D, LeakyReLU, \
    DepthwiseConv2D, MaxPooling2D as KerasMaxPool2D, AveragePooling2D as KerasAvgPool2D, LayerNormalization, Flatten
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish

from hanser.models.pooling import MaxPooling2D as MaxPool2D, AveragePooling2D as AvgPool2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization
from hanser.models.bn2 import BatchNormalizationTest
from hanser.models.evonorm import EvoNormB0, EvoNormS0
from hanser.models.modules import DropBlock, ScaledWSConv2D, AntiAliasing

__all__ = [
    "set_default", "set_defaults", "Act", "Conv2d", "Norm",
    "Linear", "GlobalAvgPool", "Pool2d", "Identity", "NormAct"]

DEFAULTS = {
    'naive_padding': False,
    'conv': {
        'depthwise': {
            'use_group': False,
            'fix_stride_with_dilation': True,
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
        'virtual_batch_size': None,
        'test': False,
    },
    'gn': {
        'groups': 32,
        'channels_per_group': 16,
        'eps': 1e-5,
        'affine': True,
    },
    'ln': {
        'eps': 1e-5,
    },
    'activation': 'relu',
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
    'dropblock': {
        'keep_prob': 0.9,
        'block_size': 7,
        'gamma_scale': 1.0,
        'per_channel': False,
    },
    'evonorm': {
        'enabled': False,
        'type': 'B0',
        'momentum': 0.9,
        'eps': 1e-5,
        'groups': 32,
    }
}

_defaults_schema = {
    'naive_padding': {'type': 'boolean'},
    'conv': {
        'depthwise': {
            'use_group': {'type': 'boolean'},
            'fix_stride_with_dilation': {'type': 'boolean'},
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
        'virtual_batch_size': {'type': 'integer', 'nullable': True},
        'test': {'type': 'boolean'},
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
    'dropblock': {
        'keep_prob': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'block_size': {'type': 'integer'},
        'gamma_scale': {'type': 'float', 'min': 0.0},
        'per_channel': {'type': 'boolean'},
    },
    'evonorm': {
        'enabled': {'type': 'boolean'},
        'type': {'type': 'string', 'allowed': ['B0', 'S0']},
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'groups': {'type': 'integer'},
    }
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


def _get_dropblock(config: Dict[str, Any]):
    return DropBlock(
        keep_prob=config['keep_prob'],
        block_size=config['block_size'],
        gamma_scale=config['gamma_scale'],
        per_channel=config['per_channel'],
    )


@curry
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
           bias_init: Optional[Initializer] = None,
           gamma_init: Union[str, Initializer] = 'ones',
           dropblock: Union[bool, Dict[str, Any]] = False,
           scaled_ws: bool = False,
           anti_alias: bool = False):

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

    if anti_alias:
        assert stride == (2, 2)
        stride = (1, 1)

    conv_cfg = DEFAULTS['conv']
    init_cfg = conv_cfg['init']
    naive_padding = DEFAULTS['naive_padding'] or padding == 'SAME'

    if naive_padding and stride != (2, 2):
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
            if conv_cfg['depthwise']['fix_stride_with_dilation'] and stride == (2, 2) and dilation == (2, 2):
                from hanser.models.conv import DepthwiseConv2D as FixedDepthwiseConv2D
                depth_conv = FixedDepthwiseConv2D
            else:
                depth_conv = DepthwiseConv2D
            conv = depth_conv(kernel_size=kernel_size, strides=stride, padding=conv_padding,
                              use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                              depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
    elif conv_cfg['group']['smart_naive'] and 1 < groups <= conv_cfg['group']['max_naive_groups']:
        conv = NaiveGroupConv2D(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=conv_padding, groups=groups)
    elif scaled_ws:
        conv = ScaledWSConv2D(
            out_channels, kernel_size=kernel_size, strides=stride,
            padding=conv_padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding=conv_padding, dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    if (not naive_padding and padding != ((0, 0), (0, 0))) or (naive_padding and stride == (2, 2)):
        conv = Sequential([
            ZeroPadding2D(padding),
            conv,
        ])

    layers = []
    if anti_alias:
        layers.append(AntiAliasing(kernel_size=3, stride=2))
    layers.append(conv)

    if DEFAULTS['evonorm']['enabled'] and norm is not None and act is not None:
        layers.append(evonorm(gamma_init))
        return Sequential(layers)

    if norm:
        layers.append(Norm(out_channels, norm, gamma_init=gamma_init))
    if dropblock:
        config = DEFAULTS['dropblock']
        if isinstance(dropblock, dict):
            config = {**config, **dropblock}
        layers.append(_get_dropblock(config))
    if act:
        layers.append(Act(act))

    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(layers)


def evonorm(gamma_init: Union[str, Initializer] = 'ones'):
    cfg = DEFAULTS['evonorm']
    if cfg['type'] == 'B0':
        norm_act = EvoNormB0(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_init)
    elif cfg['type'] == 'S0':
        norm_act = EvoNormS0(
            num_groups=cfg['groups'],
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_init)
    else:
        raise ValueError("Not reachable")
    return norm_act


def NormAct(
    channels: int,
    norm: Optional[str] = 'def',
    act: Optional[str] = 'def',
    gamma_init: Union[str, Initializer] = 'ones'):
    if DEFAULTS['evonorm']['enabled'] and norm is not None and act is not None:
        return evonorm(gamma_init)

    layers = []
    if norm:
        layers.append(Norm(channels, norm, gamma_init=gamma_init))
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


def Norm(channels=None, type='default', affine=None, track_running_stats=None, gamma_init='ones', fused=None):
    if type in ['default', 'def']:
        type = DEFAULTS['norm']
    if type == 'bn':
        cfg = DEFAULTS['bn']
        if affine is None:
            affine = cfg['affine']
        if track_running_stats is None:
            track_running_stats = cfg['track_running_stats']
        if fused is None:
            fused = cfg['fused']
        if cfg['test']:
            bn = BatchNormalizationTest(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init)
        elif cfg['sync']:
            bn = SyncBatchNormalization(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init, track_running_stats=track_running_stats,
                eval_mode=cfg['eval'])
        else:
            bn = BatchNormalization(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init, fused=fused, track_running_stats=track_running_stats,
                eval_mode=cfg['eval'], virtual_batch_size=cfg['virtual_batch_size'])
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
    elif type == 'ln':
        cfg = DEFAULTS['ln']
        return LayerNormalization(epsilon=cfg['eps'])
    elif type == 'none':
        return Identity()
    else:
        raise ValueError("Unsupported normalization type: %s" % type)


def Act(type='default', **kwargs):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'], **kwargs)
    if type == 'mish':
        return Mish()
    elif type == 'scaled_relu':
        return ScaledReLU()
    elif type == 'scaled_swish':
        return ScaledSwish()
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

    def call(self, inputs):
        return tf.identity(inputs)


def Linear(in_channels, out_channels, act=None, kernel_init=None, bias_init=None):
    kernel_initializer = kernel_init or VarianceScaling(1.0 / 3, 'fan_in', 'uniform')
    bound = math.sqrt(1 / in_channels)
    bias_initializer = bias_init or RandomUniform(-bound, bound)
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer)


def gelu(x):
    return tf.nn.sigmoid(x * 1.702) * x


class Mish(Layer):

    def call(self, x):
        return mish(x)


class ScaledReLU(Layer):

    def call(self, x):
        return tf.nn.relu(x) * 1.7139588594436646


class ScaledSwish(Layer):

    def call(self, x):
        return tf.nn.swish(x) * 1.7881293296813965


class ScaledGELU(Layer):

    def call(self, x):
        return gelu(x) * 1.7015043497085571


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
        xs = tf.split(x, self.groups, axis=-1)
        xs = [
            conv(x) for conv, x in zip(self.convs, xs)
        ]
        x = tf.concat(xs, axis=-1)
        return x
