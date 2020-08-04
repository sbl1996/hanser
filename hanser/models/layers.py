from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.layers import Dense, DepthwiseConv2D, Activation, AvgPool2D, MaxPool2D, Layer, InputSpec
from tensorflow.keras.regularizers import l2
from tensorflow_addons.activations import mish

from hanser.models.bn import BatchNormalization, SyncBatchNormalization
from hanser.models.conv import Conv2D

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
    'weight_decay': None,
}


def set_default(keys: Union[str, Sequence[str]], value):
    def loop(d, keys):
        k = keys[0]
        if k not in d:
            match = get_close_matches(k, d.keys())
            if match:
                raise KeyError("No such key `%s`, maybe you mean `%s`" % (k, match[0]))
            else:
                raise KeyError("No key `%s` in %s" % (k, d))
        if len(keys) == 1:
            d[k] = value
        else:
            loop(d[k], keys[1:])

    if isinstance(keys, str):
        keys = [keys]
    loop(DEFAULTS, keys)


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
    elif bias is None and norm:
        use_bias = False
    else:
        use_bias = bias

    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    if (norm or act):
        conv_name = 'conv'
    else:
        conv_name = name
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
    if not (norm or act):
        return conv
    layers = [conv]
    if norm:
        layers.append(Norm(out_channels, norm, zero_init=zero_init, name="norm"))
    if act:
        layers.append(Act(act, name="act"))
    return Sequential(layers, name=name)


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
    track_running_stats = track_running_stats or cfg['track_running_stats']
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
