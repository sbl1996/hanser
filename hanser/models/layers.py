import math
from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping

from cerberus import Validator

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomNormal, RandomUniform
from tensorflow.keras.layers import Dense, Activation, Layer, InputSpec, Conv2D, ZeroPadding2D, LeakyReLU, \
    Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow_addons.activations import mish as tfa_mish

from hanser.models.pooling import MaxPooling2D as MaxPool2D, AveragePooling2D as AvgPool2D
from hanser.models.conv import DepthwiseConv2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization

__all__ = ["set_default", "set_defaults", "Act", "Conv2d", "Norm", "Linear", "GlobalAvgPool", "Pool2d", "Identity"]

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
    'compat': False,
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
    'init': {
        'type': 'msra',
        'mode': 'fan_in',
        'uniform': True,
        'std': 0.01,
        'scale': 1.0,
    },
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
    'activation': {'type': 'string', 'allowed': ['relu', 'swish', 'mish', 'leaky_relu', 'sigmoid']},
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn']},
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

    kernel_regularizer = get_weight_decay()
    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    if in_channels == groups:
        depth_multiplier = out_channels // in_channels
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='valid',
                               use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                               depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               depthwise_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    else:
        conv = Conv2D(out_channels, kernel_size=kernel_size, strides=stride,
                      padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

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

    kernel_regularizer = get_weight_decay()
    bias_regularizer = get_weight_decay() if not DEFAULTS['no_bias_decay'] else None

    if in_channels == groups:
        depth_multiplier = out_channels // in_channels
        conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='valid',
                               use_bias=use_bias, dilation_rate=dilation, depth_multiplier=depth_multiplier,
                               depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               depthwise_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    else:
        conv = Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=stride,
                               padding='valid', dilation_rate=dilation, use_bias=use_bias, groups=groups,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

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


def Norm(channels, type='default', affine=None, track_running_stats=None):
    if type in ['default', 'def']:
        type = DEFAULTS['norm']
    assert type == 'bn'
    cfg = DEFAULTS['bn']
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
            center=affine, scale=affine, track_running_stats=track_running_stats)
    else:
        bn = BatchNormalization(
            momentum=cfg['momentum'], epsilon=cfg['eps'], gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer, beta_regularizer=beta_regularizer, fused=cfg['fused'],
            center=affine, scale=affine, track_running_stats=track_running_stats)
    return bn


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


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=False, name=None):
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


def get_weight_decay():
    wd = DEFAULTS['weight_decay']
    if wd:
        return l2(wd * 0.5)


def Linear(in_channels, out_channels, act=None, name=None):
    kernel_initializer = VarianceScaling(1.0 / 3, 'fan_in', 'uniform')
    bound = math.sqrt(1 / in_channels)
    bias_initializer = RandomUniform(-bound, bound)
    return Dense(out_channels, activation=act,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=get_weight_decay(),
                 bias_regularizer=get_weight_decay())


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


# class VarianceScaling(Initializer):
#   """Initializer capable of adapting its scale to the shape of weights tensors.
#
#   Initializers allow you to pre-specify an initialization strategy, encoded in
#   the Initializer object, without knowing the shape and dtype of the variable
#   being initialized.
#
#   With `distribution="truncated_normal" or "untruncated_normal"`, samples are
#   drawn from a truncated/untruncated normal distribution with a mean of zero and
#   a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`
#   where n is:
#
#     - number of input units in the weight tensor, if mode = "fan_in"
#     - number of output units, if mode = "fan_out"
#     - average of the numbers of input and output units, if mode = "fan_avg"
#
#   With `distribution="uniform"`, samples are drawn from a uniform distribution
#   within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
#
#   Examples:
#
#   >>> def make_variables(k, initializer):
#   ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
#   ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
#   >>> v1, v2 = make_variables(3, tf.initializers.VarianceScaling(scale=1.))
#   >>> v1
#   <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
#   >>> v2
#   <tf.Variable ... shape=(3, 3) ... numpy=
#   ...
#   >>> make_variables(4, tf.initializers.VarianceScaling(distribution='uniform'))
#   (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
#
#   Args:
#     scale: Scaling factor (positive float).
#     mode: One of "fan_in", "fan_out", "fan_avg".
#     distribution: Random distribution to use. One of "truncated_normal",
#       "untruncated_normal" and  "uniform".
#     seed: A Python integer. Used to create random seeds. See
#       `tf.random.set_seed` for behavior.
#
#   Raises:
#     ValueError: In case of an invalid value for the "scale", mode" or
#       "distribution" arguments.
#   """
#
#   def __init__(self):
#     if scale <= 0.:
#       raise ValueError("`scale` must be positive float.")
#     if mode not in {"fan_in", "fan_out", "fan_avg"}:
#       raise ValueError("Invalid `mode` argument:", mode)
#     distribution = distribution.lower()
#     # Compatibility with keras-team/keras.
#     if distribution == "normal":
#       distribution = "truncated_normal"
#     if distribution not in {"uniform", "truncated_normal",
#                             "untruncated_normal"}:
#       raise ValueError("Invalid `distribution` argument:", distribution)
#     self.scale = scale
#     self.mode = mode
#     self.distribution = distribution
#     self.seed = seed
#     self._random_generator = _RandomGenerator(seed)
#
#   def __call__(self, shape, dtype=dtypes.float32):
#     """Returns a tensor object initialized as specified by the initializer.
#
#     Args:
#       shape: Shape of the tensor.
#       dtype: Optional dtype of the tensor. Only floating point types are
#        supported.
#
#     Raises:
#       ValueError: If the dtype is not floating point
#     """
#     partition_info = None  # Keeps logic so can be readded later if necessary
#     dtype = _assert_float_dtype(dtype)
#     scale = self.scale
#     scale_shape = shape
#     if partition_info is not None:
#       scale_shape = partition_info.full_shape
#     fan_in, fan_out = _compute_fans(scale_shape)
#     if self.mode == "fan_in":
#       scale /= max(1., fan_in)
#     elif self.mode == "fan_out":
#       scale /= max(1., fan_out)
#     else:
#       scale /= max(1., (fan_in + fan_out) / 2.)
#     if self.distribution == "truncated_normal":
#       # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
#       stddev = math.sqrt(scale) / .87962566103423978
#       return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
#     elif self.distribution == "untruncated_normal":
#       stddev = math.sqrt(scale)
#       return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
#     else:
#       limit = math.sqrt(3.0 * scale)
#       return self._random_generator.random_uniform(shape, -limit, limit, dtype)
#
#   def get_config(self):
#     return {
#         "scale": self.scale,
#         "mode": self.mode,
#         "distribution": self.distribution,
#         "seed": self.seed
#     }