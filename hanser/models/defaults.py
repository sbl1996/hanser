from typing import Mapping, Union, Sequence
from difflib import get_close_matches
from cerberus import Validator

DEFAULTS = {
    'inplace_abn': {
        'enabled': False,
        'momentum': 0.9,
        'eps': 1.001e-5,
        'alpha': 0.01,
        'sync': False,
        'fused': None,
    },
    'fixed_padding': False,
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
        'per_channel': True,
    },
    'evonorm': {
        'enabled': False,
        'type': 'B0',
        'momentum': 0.9,
        'eps': 1e-5,
        'groups': 32,
    },
    'anti_aliasing': {
        'mode': "CONSTANT",
    }
}

_defaults_schema = {
    'inplace_abn': {
        'enabled': {'type': 'boolean'},
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'alpha': {'type': 'float', 'min': 0.0},
        'sync': {'type': 'boolean'},
        'fused': {'type': 'boolean', 'nullable': True},
    },
    'fixed_padding': {'type': 'boolean'},
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
