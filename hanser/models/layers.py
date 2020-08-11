from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping

from cerberus import Validator

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.layers import Dense, Activation, AvgPool2D, MaxPool2D, Layer, InputSpec, Conv2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow_addons.activations import mish

from hanser.models.conv import DepthwiseConv2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization

__all__ = ["set_default", "Act", "Conv2d", "Norm", "Linear", "GlobalAvgPool", "Pool2d"]

DEFAULTS = {
    'tpu': False,
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
        'fused': True,
        'sync': False,
    },
    'activation': 'relu',
    'norm': 'bn',
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
    'weight_decay': 0.0,
}

_defaults_schema = {
    'tpu': {'type': 'boolean'},
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
        'fused': {'type': 'boolean'},
        'sync': {'type': 'boolean'},
    },
    'activation': {'type': 'string', 'allowed': ['relu', 'swish', 'mish']},
    'norm': {'type': 'string', 'allowed': ['bn']},
    'fp16': {'type': 'boolean'},
    'init': {
        'type': {'type': 'string', 'allowed': ['msra', 'normal']},
        'mode': {'type': 'string', 'allowed': ['fan_in', 'fan_out']},
        'uniform': {'type': 'boolean'},
        'std': {'type': 'float', 'min': 0.0},
        'scale': {'type': 'float', 'min': 0.0},
    },
    'seed': {'type': 'integer'},
    'no_bias_decay': {'type': 'boolean'},
    'weight_decay': {'type': 'float', 'min': 0.0, 'max': 1.0},
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


class Padding2D(Layer):

    def __init__(self, padding, mode='CONSTANT', constant_values=0, name=None):
        super().__init__(name=name)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            if isinstance(padding[0], int):
                height_padding = (padding[0], padding[0])
            else:
                height_padding = padding[0]
            if isinstance(padding[1], int):
                width_padding = (padding[1], padding[1])
            else:
                width_padding = padding[1]
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.mode = mode
        self.constant_values = constant_values

    def call(self, x, training=None):
        return tf.pad(x, [(0, 0), *self.padding, (0, 0)], self.mode, self.constant_values)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if input_shape[1] is not None:
            rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
        else:
            rows = None
        if input_shape[2] is not None:
            cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
        else:
            cols = None
        return tf.TensorShape(
            [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = {'padding': self.padding, 'mode': self.mode, 'constant_values': self.constant_values}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = (ph, pw)
    return padding


def Conv2d(in_channels: int,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[str] = None,
           zero_init=False,
           name: Optional[str] = None):
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
    elif bias is None and norm:
        use_bias = False
    else:
        use_bias = bias

    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    def make_conv(name):
        if in_channels == groups:
            depth_multiplier = out_channels // in_channels
            conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='valid',
                                   use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                                   depthwise_initializer=kernel_initializer, bias_initializer='zeros',
                                   kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer, name=name)
        else:
            conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                          padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                          kernel_initializer=kernel_initializer, bias_initializer='zeros',
                          kernel_regularizer=get_weight_decay(), bias_regularizer=bias_regularizer, name=name)
        return conv

    if not (norm or act):
        if padding[0] != 0:
            conv = Sequential([
                ZeroPadding2D(padding, name='conv_pad'),
                make_conv("conv"),
            ], name=name)
        else:
            conv = make_conv(name)
        return conv
    else:
        if padding[0] != 0:
            conv = Sequential([
                ZeroPadding2D(padding, name='conv_pad'),
                make_conv("conv"),
            ], name="conv")
        else:
            conv = make_conv("conv")
        layers = [conv]
        if norm:
            layers.append(Norm(out_channels, norm, zero_init=zero_init, name="norm"))
        if act:
            layers.append(Act(act, name="act"))
        return Sequential(layers, name=name)


# noinspection PyUnusedLocal
def Norm(channels, type='default', affine=None, track_running_stats=None, zero_init=False, name=None):
    if type in ['default', 'def']:
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
    if affine is None:
        affine = cfg['affine']
    if track_running_stats is None:
        track_running_stats = cfg['track_running_stats']
    if cfg['sync']:
        bn = SyncBatchNormalization(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer,
            center=affine, scale=affine, track_running_stats=track_running_stats, name=name)
    else:
        bn = BatchNormalization(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer, fused=cfg['fused'],
            center=affine, scale=affine, track_running_stats=track_running_stats, name=name)
    return bn


def Act(type='default', name=None):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'], name)
    elif type == 'mish':
        if DEFAULTS['tpu']:
            return CustomMish(name)
        else:
            return Mish(name)
    else:
        return Activation(type, name=name)


# noinspection PyUnusedLocal
def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=False, name=None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding == 'same':
        padding = calc_same_padding(kernel_size, (1, 1))

    if type == 'avg':
        pool = AvgPool2D
    elif type == 'max':
        pool = MaxPool2D
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    if padding[0] == 0:
        return pool(kernel_size, stride, 'valid', name=name)
    else:
        return Sequential([
            Padding2D(padding, mode='SYMMETRIC', name='pool_pad'),
            pool(kernel_size, stride, 'valid', name="pool"),
        ], name=name)


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


def get_weight_decay():
    wd = DEFAULTS['weight_decay']
    if wd:
        return l2(wd)


# noinspection PyUnusedLocal
def Linear(in_channels, out_channels, act=None, name=None):
    kernel_initializer = RandomNormal(0, 0.01, seed=DEFAULTS['seed'])
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=get_weight_decay(),
                 bias_regularizer=get_weight_decay(),
                 name=name)


class Mish(Layer):

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, x, training=None):
        return mish(x)


def custom_mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


class CustomMish(Layer):

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, x, training=None):
        return custom_mish(x)
