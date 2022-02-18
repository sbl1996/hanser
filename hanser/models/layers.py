import math
from typing import Union, Tuple, Optional, Dict, Any
from toolz import curry

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Initializer
from tensorflow.keras.layers import Dense, Activation, Conv2D, ZeroPadding2D, LeakyReLU, \
    DepthwiseConv2D, MaxPooling2D as KerasMaxPool2D, AveragePooling2D as KerasAvgPool2D, LayerNormalization

from hanser.models.pooling import MaxPooling2D as MaxPool2D, AveragePooling2D as AvgPool2D
from hanser.models.bn import BatchNormalization, SyncBatchNormalization
from hanser.models.bn2 import BatchNormalizationTest
from hanser.models.inplace_abn import InplaceABN
from hanser.models.evonorm import EvoNormB0, EvoNormS0
from hanser.models.modules import DropBlock, ScaledWSConv2D, AntiAliasing, GlobalAvgPool, Identity, NaiveGroupConv2D, \
    GELU, Mish, ScaledSwish, ScaledGELU, ScaledReLU, Dropout, ReLU6
from hanser.models.defaults import DEFAULTS, set_defaults, set_default


__all__ = [
    "set_default", "set_defaults", "Act", "Conv2d", "Norm",
    "Linear", "GlobalAvgPool", "Pool2d", "Identity", "NormAct", "Dropout"]


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = ((ph, ph), (pw, pw))
    return padding


def calc_fixed_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = kh + (kh - 1) * (dh - 1) - 1
    ph_1, ph_2 = ph // 2, ph - ph // 2
    pw = kw + (kw - 1) * (dw - 1) - 1
    pw_1, pw_2 = pw // 2, pw - pw // 2
    padding = ((ph_1, ph_2), (pw_1, pw_2))
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
           avd=False, avd_first=True,
           anti_alias: bool = False):

    assert not (avd and anti_alias)

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
    elif isinstance(padding, str):
        assert padding == 'same'

    # assert stride in [(1, 1), (2, 2)]

    avd = avd and stride == (2, 2)
    anti_alias = anti_alias and stride == (2, 2)

    if avd or anti_alias:
        assert norm is not None and act is not None
        stride = (1, 1)

    conv_cfg = DEFAULTS['conv']
    init_cfg = conv_cfg['init']


    # There are 4 types of padding mode:
    if padding == 'same':
        # 1. fixed padding
        if DEFAULTS['fixed_padding']:
            if 2 in stride:
                paddings = calc_fixed_padding(kernel_size, dilation)
                pad = ZeroPadding2D(paddings) if paddings != ((0, 0), (0, 0)) else None
                conv_padding = 'VALID'
            else:
                pad = None
                if kernel_size == (1, 1):
                    conv_padding = 'VALID'
                else:
                    conv_padding = 'SAME'
        # 2. naive padding
        elif DEFAULTS['naive_padding']:
            pad = None
            if kernel_size == (1, 1):
                conv_padding = 'VALID'
            else:
                conv_padding = 'SAME'
        # 3. same padding (hanser previously used)
        else:
            paddings = calc_same_padding(kernel_size, dilation)
            if paddings == ((0, 0), (0, 0)):
                pad = None
            else:
                pad = ZeroPadding2D(paddings)
            conv_padding = 'VALID'
    else:
        # 4. manual padding
        if padding != ((0, 0), (0, 0)):
            pad = ZeroPadding2D(padding)
        else:
            pad = None
        conv_padding = 'VALID'


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
    elif init_cfg['zero_bias']:
        bias_initializer = 'zeros'
    else:
        bound = math.sqrt(1 / (kernel_size[0] * kernel_size[1] * (in_channels // groups)))
        bias_initializer = RandomUniform(-bound, bound)

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    if in_channels == groups and in_channels != 1:
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

    if pad is not None:
        conv = Sequential([pad, conv])

    layers = [conv]
    if avd and avd_first:
        layers.insert(0, Pool2d(kernel_size=3, stride=2, type='avg'))

    if DEFAULTS['evonorm']['enabled'] and norm is not None and act is not None:
        layers.append(evonorm(gamma_init))
    elif DEFAULTS['inplace_abn']['enabled'] and norm is not None and act is not None:
        layers.append(inplace_abn(gamma_init))
    else:
        if norm:
            layers.append(Norm(out_channels, norm, gamma_init=gamma_init))
        if dropblock:
            config = DEFAULTS['dropblock']
            if isinstance(dropblock, dict):
                config = {**config, **dropblock}
            layers.append(_get_dropblock(config))
        if act:
            layers.append(Act(act))

    if avd and not avd_first:
        layers.append(Pool2d(kernel_size=3, stride=2, type='avg'))

    if anti_alias:
        cfg = DEFAULTS['anti_aliasing']
        layers.append(
            AntiAliasing(kernel_size=cfg['kernel_size'], stride=2,
                         mode=cfg['mode'], learnable=cfg['learnable']))

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


def inplace_abn(gamma_init: Union[str, Initializer] = 'ones'):
    cfg = DEFAULTS['inplace_abn']
    norm_act = InplaceABN(
        momentum=cfg['momentum'], epsilon=cfg['eps'], alpha=cfg['alpha'],
        sync=cfg['sync'], fused=cfg['fused'], gamma_initializer=gamma_init)
    return norm_act


def NormAct(
    channels: int,
    norm: Optional[str] = 'def',
    act: Optional[str] = 'def',
    gamma_init: Union[str, Initializer] = 'ones'):
    if DEFAULTS['evonorm']['enabled'] and norm is not None and act is not None:
        return evonorm(gamma_init)
    elif DEFAULTS['inplace_abn']['enabled'] and norm is not None and act is not None:
        return inplace_abn(gamma_init)
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
        from tensorflow_addons.layers import GroupNormalization

        try:
            from tensorflow_addons.layers import GroupNormalization
        except ImportError:
            raise ImportError(
                "Please install tensorflow_addons:'pip install -U tensorflow_addons==0.13.0'")
        gn = GroupNormalization(
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
    elif type == 'gelu':
        return GELU(approximate=DEFAULTS['gelu']['approximate'])
    elif type == 'relu6':
        return ReLU6()

    elif type == 'scaled_relu':
        return ScaledReLU()
    elif type == 'scaled_swish':
        return ScaledSwish()
    elif type == 'scaled_gelu':
        return ScaledGELU()

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

    horch_impl = not (DEFAULTS['naive_padding'] or DEFAULTS['fixed_padding'])

    if type == 'avg':
        pool = AvgPool2D if horch_impl else KerasAvgPool2D
    elif type == 'max':
        pool = MaxPool2D if horch_impl else KerasMaxPool2D
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    return pool(kernel_size, stride, padding)


def Linear(in_channels, out_channels, act=None, bias=True, kernel_init=None, bias_init=None):
    kernel_initializer = kernel_init or VarianceScaling(1.0 / 3, 'fan_in', 'uniform')
    bound = math.sqrt(1 / in_channels)
    bias_initializer = bias_init or RandomUniform(-bound, bound)
    return Dense(out_channels, activation=act, use_bias=bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer)