# These implementations were stolen from https://github.com/google/automl/blob/master/efficientnetv2/autoaugment.py
# 1. Image ops are exactly the same.
# 2. Function to combine and apply policies is different.
# 3. The way to apply hparams (cutout_max and translate_max) is different.

from typing import Optional, Dict

import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, solarize_add, autocontrast, translate_x, \
    translate_y, rotate, color, posterize, contrast, brightness, equalize, invert, cutout2 as cutout, random_apply
from hanser.transform.common import image_dimensions


H_PARAMS = {
    "max_level": 10,
    "fill_color": (128, 128, 128),
    # Follows paper rather than code
    'cutout_max': 60 / 331,
    'translate_max': 150 / 331.,
    'rotate_max': 30.,
    'enhance_min': 0.1,
    'enhance_max': 1.9,
    'posterize_min': 4.0,
    'posterize_max': 8.0,
    'solarize_max': 256,
    'solarize_add_max': 110,
    'shear_max': 0.3,
}


def _randomly_negate_tensor(tensor):
    tensor = tf.convert_to_tensor(tensor)
    sign = tf.sign(tf.random.normal(()))
    sign = tf.convert_to_tensor(sign, tensor.dtype)
    return tensor * sign


def _rotate_level_to_arg(level, max_level, max_val):
    level = level / max_level * max_val
    level = _randomly_negate_tensor(level)
    return level


def _enhance_level_to_arg(level, max_level, min_val, max_val):
    return (level / max_level) * (max_val - min_val) + min_val


def _translate_level_to_arg(level, max_level, max_val):
    level = (level / max_level) * max_val
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level, max_level, min_val, max_val):
    level = max_val - (level / max_level) * (max_val - min_val)
    level = tf.cast(level, tf.int32)
    return level


def _solarize_level_to_arg(level, max_level, max_val):
    level = tf.cast((level / max_level) * max_val, tf.int32)
    return level


def _solarize_add_level_to_arg(level, max_level, max_val):
    level = tf.cast((level / max_level) * max_val, tf.int32)
    return level


def _cutout_level_to_arg(level, max_level, max_val):
    return (level / max_level) * max_val


def _shear_level_to_arg(level, max_level, max_val):
    level = (level / max_level) * max_val
    level = _randomly_negate_tensor(level)
    return level


def _shear_x(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _shear_level_to_arg(level, hparams['max_level'], hparams['shear_max'])
    return shear_x(img, magnitude, fill_color)


def _shear_y(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _shear_level_to_arg(level, hparams['max_level'], hparams['shear_max'])
    return shear_y(img, magnitude, fill_color)


def _translate_x(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _translate_level_to_arg(level, hparams['max_level'], hparams['translate_max'])
    magnitude = tf.cast(tf.shape(img)[1], magnitude.dtype) * magnitude
    return translate_x(img, magnitude, fill_color)


def _translate_y(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _translate_level_to_arg(level, hparams['max_level'], hparams['translate_max'])
    magnitude = tf.cast(tf.shape(img)[0], magnitude.dtype) * magnitude
    return translate_y(img, magnitude, fill_color)


def _rotate(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _rotate_level_to_arg(level, hparams['max_level'], hparams['rotate_max'])
    return rotate(img, magnitude, fill_color)


def _posterize(img, level, hparams):
    magnitude = _posterize_level_to_arg(
        level, hparams['max_level'], hparams['posterize_min'], hparams['posterize_max'])
    return posterize(img, magnitude)


def _solarize(img, level, hparams):
    magnitude = _solarize_level_to_arg(level, hparams['max_level'], hparams['solarize_max'])
    return solarize(img, magnitude)


def _solarize_add(img, level, hparams):
    magnitude = _solarize_add_level_to_arg(level, hparams['max_level'], hparams['solarize_add_max'])
    return solarize_add(img, magnitude)


def _color(img, level, hparams):
    magnitude = _enhance_level_to_arg(
        level, hparams['max_level'], hparams['enhance_min'], hparams['enhance_max'])
    return color(img, magnitude)


def _contrast(img, level, hparams):
    magnitude = _enhance_level_to_arg(
        level, hparams['max_level'], hparams['enhance_min'], hparams['enhance_max'])
    return contrast(img, magnitude)


def _sharpness(img, level, hparams):
    magnitude = _enhance_level_to_arg(
        level, hparams['max_level'], hparams['enhance_min'], hparams['enhance_max'])
    return sharpness(img, magnitude)


def _brightness(img, level, hparams):
    magnitude = _enhance_level_to_arg(
        level, hparams['max_level'], hparams['enhance_min'], hparams['enhance_max'])
    return brightness(img, magnitude)


# noinspection PyUnusedLocal
def _autocontrast(img, level, hparams):
    return autocontrast(img)


# noinspection PyUnusedLocal
def _equalize(img, level, hparams):
    return equalize(img)


# noinspection PyUnusedLocal
def _invert(img, level, hparams):
    return invert(img)


def _cutout(img, level, hparams):
    fill_color = hparams['fill_color']
    magnitude = _cutout_level_to_arg(level, hparams['max_level'], hparams['cutout_max'])
    h, w = image_dimensions(img, 3)[:2]
    magnitude = tf.cast(tf.cast(tf.maximum(h, w), tf.float32) * magnitude, tf.int32)
    return cutout(img, magnitude, fill_color)


# noinspection PyUnusedLocal
def _identity(img, level, hparams):
    return img


NAME_TO_FUNC = {
    "identity": _identity,
    "autocontrast": _autocontrast,
    "equalize": _equalize,
    "rotate": _rotate,
    "solarize": _solarize,
    "color": _color,
    "posterize": _posterize,
    "contrast": _contrast,
    "brightness": _brightness,
    "sharpness": _sharpness,

    "shearX": _shear_x,
    "shearY": _shear_y,
    'translateX': _translate_x,
    'translateY': _translate_y,

    "solarize_add": _solarize_add,
    "invert": _invert,
    'cutout': _cutout,
}


def _apply_func_with_prob(func, p, image, level, hparams):
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + p), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, level, hparams),
        lambda: image)
    return augmented_image


def sub_policy(p1, op1, level1, p2, op2, level2, hparams):
    def _apply_policy(image):
        image = _apply_func_with_prob(NAME_TO_FUNC[op1], p1, image, level1, hparams)
        image = _apply_func_with_prob(NAME_TO_FUNC[op2], p2, image, level2, hparams)
        return image
    return _apply_policy


def apply_autoaugment(image, policies, hparams: Optional[Dict]=None):
    hparams = {
        **H_PARAMS,
        **(hparams or {}),
    }

    policy_to_select = tf.random.uniform((), maxval=len(policies), dtype=tf.int32)
    for i, policy in enumerate(policies):
        policy_fn = sub_policy(*policy, hparams)
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda: policy_fn(image),
            lambda: image)
    return image
