import math
from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping

from cerberus import Validator

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomNormal, RandomUniform
from tensorflow.keras.layers import Dense, Activation, Layer, InputSpec, Conv2D, ZeroPadding2D, LeakyReLU, \
    Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish as tfa_mish

from hanser.models.pooling import MaxPooling2D as MaxPool2D, AveragePooling2D as AvgPool2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization

__all__ = ["set_default", "set_defaults", "Act", "Conv2d", "Norm", "Linear", "GlobalAvgPool", "Pool2d", "Identity"]

DEFAULTS = {
    'conv': {
        'depthwise': {
            'use_group': False,
            'horch': False,
        }
    },
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
        'fused': True,
        'sync': False,
    },
    'gn': {
        'groups': None,
        'channels_per_group': 16,
        'eps': 1e-5,
        'affine': True,
    },
    'activation': 'relu',
    'compat': False,
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
    'init': {
        'type': 'msra',
        'mode': 'fan_out',
        'distribution': 'untruncated_normal',
        'fix': True,
    },
}

_defaults_schema = {
    'conv': {
        'depthwise': {
            'use_group': {'type': 'boolean'},
            'horch': {'type': 'boolean'},
        }
    },
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
        'fused': {'type': 'boolean'},
        'sync': {'type': 'boolean'},
    },
    'gn': {
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'groups': {'type': 'integer'},
        'channels_per_group': {'type': 'integer'},
    },
    'activation': {'type': 'string', 'allowed': ['relu', 'swish', 'mish', 'leaky_relu', 'sigmoid']},
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn', 'gn', 'none']},
    'init': {
        'type': {'type': 'string', 'allowed': ['msra', 'normal']},
        'mode': {'type': 'string', 'allowed': ['fan_in', 'fan_out']},
        'distribution': {'type': 'string', 'allowed': ['uniform', 'truncated_normal','untruncated_normal']},
        'fix': {'type': 'boolean'},
    },
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
    loop(DEFAULTS, keys, _defaults_schema)


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = (ph, pw)
    return padding


def flip_mode(m):
    if m == 'fan_in':
        return 'fan_out'
    else:
        return 'fan_in'


def Conv2d(in_channels: int,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[str] = None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(padding, str):
        assert padding == 'same'
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    # Init
    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        mode = init_cfg['mode']
        distribution = init_cfg['distribution']
        if in_channels == groups and not DEFAULTS['conv']['depthwise']['use_group'] and DEFAULTS['init']['fix']:
            mode = flip_mode(mode)
        if 'uniform' in distribution:
            kernel_initializer = VarianceScaling(1.0 / 3, mode, distribution)
        else:
            kernel_initializer = VarianceScaling(2.0, mode, distribution)
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    bound = math.sqrt(1 / (kernel_size[0] * kernel_size[1] * (in_channels // groups)))
    bias_initializer = RandomUniform(-bound, bound)

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    if in_channels == groups:
        if DEFAULTS['conv']['depthwise']['use_group']:
            conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                          padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        else:
            depth_multiplier = out_channels // in_channels
            if DEFAULTS['conv']['depthwise']['horch']:
                from hanser.models.conv import DepthwiseConv2D as HorchDepthwiseConv2D
                depth_conv = HorchDepthwiseConv2D
            else:
                depth_conv = DepthwiseConv2D
            conv = depth_conv(kernel_size=kernel_size, strides=stride, padding='valid',
                              use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                              depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    if padding != (0, 0):
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


def ConvTranspose2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 'same',
    groups: int = 1,
    dilation: int = 1,
    bias: Optional[bool] = None,
    norm: Optional[str] = None,
    act: Optional[str] = None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(padding, str):
        assert padding == 'same'
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    # Init
    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        if init_cfg['uniform']:
            kernel_initializer = VarianceScaling(
                1.0 / 3 * init_cfg['scale'], init_cfg['mode'], 'uniform')
        else:
            kernel_initializer = VarianceScaling(
                2.0 * init_cfg['scale'], init_cfg['mode'], 'untruncated_normal')
    elif init_cfg['type'] == 'normal':
        kernel_initializer = RandomNormal(0, init_cfg['std'])
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    bound = math.sqrt(1 / (kernel_size[0] * kernel_size[1] * in_channels))
    bias_initializer = RandomUniform(-bound, bound)

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    if in_channels == groups:
        depth_multiplier = out_channels // in_channels
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='valid',
                               use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                               depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
    else:
        conv = Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=stride,
                               padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    if padding != (0, 0):
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
                gamma_initializer=gamma_init, track_running_stats=track_running_stats)
        else:
            bn = BatchNormalization(
                momentum=cfg['momentum'], epsilon=cfg['eps'], center=affine, scale=affine,
                gamma_initializer=gamma_init, fused=cfg['fused'], track_running_stats=track_running_stats)
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

def Act(type='default'):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'])
    if type == 'mish':
        return Mish(compat=DEFAULTS['compat'])
    elif type == 'swish' and DEFAULTS['compat']:
        return Swish()
    elif type == 'leaky_relu':
        return LeakyReLU(alpha=DEFAULTS['leaky_relu']['alpha'])
    else:
        return Activation(type)


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=False):
    assert padding == 0 or padding == 'same'
    if padding == 0:
        padding = 'valid'

    if type == 'avg':
        pool = AvgPool2D
    elif type == 'max':
        pool = MaxPool2D
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    return pool(kernel_size, stride, padding)


class GlobalAvgPool(Layer):
    """Abstract class for different global pooling 2D layers.
  """

    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim
        self.input_spec = InputSpec(ndim=4)
        self._supports_ragged_inputs = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.keep_dim:
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])
        else:
            return tf.TensorShape([input_shape[0], input_shape[3]])

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


def Linear(in_channels, out_channels, act=None):
    kernel_initializer = VarianceScaling(1.0 / 3, 'fan_in', 'uniform')
    bound = math.sqrt(1 / in_channels)
    bias_initializer = RandomUniform(-bound, bound)
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


class Mish(Layer):

    def __init__(self, compat=False, **kwargs):
        self.compat = compat
        super().__init__(**kwargs)

    def call(self, x, training=None):
        if self.compat:
            return mish(x)
        else:
            return tfa_mish(x)

    def get_config(self):
        base_config = super().get_config()
        return base_config


class Swish(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        return tf.nn.swish(x)

    def get_config(self):
        base_config = super().get_config()
        return base_config