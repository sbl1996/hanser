import math

import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, autocontrast, translate_x, translate_y, rotate, color, posterize, contrast, brightness, equalize, invert

_MAX_LEVEL = 10
_FILL_COLOR = (128, 128, 128)


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random_uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level


def _enhance_level_to_arg(level):
    # TODO: To complex control flow, may be fixed in TF 2.0
    # level = (level / _MAX_LEVEL) * 0.9
    # level = _randomly_negate_tensor(level)
    # return level
    return (level / _MAX_LEVEL) * 1.8 + 0.1


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * (32 * 150 / 331)
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level):
    level = (8, 8, 7, 7, 6, 6, 5, 5, 4, 4)[level]
    return level


def _solarize_level_to_arg(level):
    level = (_MAX_LEVEL - level + 1) / _MAX_LEVEL * 256
    return level


NAME_TO_FUNC = {
    "shearX": lambda img, level: shear_x(
        img, _shear_level_to_arg(level), _FILL_COLOR),
    "shearY": lambda img, level: shear_y(
        img, _shear_level_to_arg(level), _FILL_COLOR),
    'translateX': lambda img, level: translate_x(
        img, _translate_level_to_arg(level), _FILL_COLOR),
    'translateY': lambda img, level: translate_y(
        img, _translate_level_to_arg(level), _FILL_COLOR),
    "rotate": lambda img, level: rotate(
        img, _rotate_level_to_arg(level), _FILL_COLOR),
    "color": lambda img, level: color(
        img, _enhance_level_to_arg(level)),
    "posterize": lambda img, level: posterize(img, level),
    "solarize": lambda img, level: solarize(img, level),
    "contrast": lambda img, level: contrast(
        img, _enhance_level_to_arg(level)),
    "sharpness": lambda img, level: sharpness(
        img, _enhance_level_to_arg(level)),
    "brightness": lambda img, level: brightness(
        img, _enhance_level_to_arg(level)),
    "autocontrast": lambda img, level: autocontrast(img),
    "equalize": lambda img, level: equalize(img),
    "invert": lambda img, level: invert(img),
}


def _apply_func_with_prob(func, p, image, level):
    should_apply_op = tf.cast(
        tf.floor(tf.random_uniform([], dtype=tf.float32) + p), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, level),
        lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""

    policy_to_select = tf.random_uniform((), maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(policies):
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda: policy(image),
            lambda: image)
    return image


def sub_policy(p1, op1, level1, p2, op2, level2):
    def _apply_policy(image):
        image = _apply_func_with_prob(NAME_TO_FUNC[op1], p1, image, level1)
        image = _apply_func_with_prob(NAME_TO_FUNC[op2], p2, image, level2)
        return image

    return _apply_policy


def cifar10_policy():
    policies = [
        sub_policy(0.1, "invert", 7, 0.2, "contrast", 6),
        sub_policy(0.7, "rotate", 2, 0.3, "translateX", 9),
        sub_policy(0.8, "sharpness", 1, 0.9, "sharpness", 3),
        sub_policy(0.5, "shearY", 8, 0.7, "translateY", 9),
        sub_policy(0.5, "autocontrast", 8, 0.9, "equalize", 2),

        sub_policy(0.2, "shearY", 7, 0.3, "posterize", 7),
        sub_policy(0.4, "color", 3, 0.6, "brightness", 7),
        sub_policy(0.3, "sharpness", 9, 0.7, "brightness", 9),
        sub_policy(0.6, "equalize", 5, 0.5, "equalize", 1),
        sub_policy(0.6, "contrast", 7, 0.6, "sharpness", 5),

        sub_policy(0.7, "color", 7, 0.5, "translateX", 8),
        sub_policy(0.3, "equalize", 7, 0.4, "autocontrast", 8),
        sub_policy(0.4, "translateY", 3, 0.2, "sharpness", 6),
        sub_policy(0.9, "brightness", 6, 0.2, "color", 8),
        sub_policy(0.5, "solarize", 2, 0.0, "invert", 3),

        sub_policy(0.2, "equalize", 0, 0.6, "autocontrast", 0),
        sub_policy(0.2, "equalize", 8, 0.8, "equalize", 4),
        sub_policy(0.9, "color", 9, 0.6, "equalize", 6),
        sub_policy(0.8, "autocontrast", 4, 0.2, "solarize", 8),
        sub_policy(0.1, "brightness", 3, 0.7, "color", 0),

        sub_policy(0.4, "solarize", 5, 0.9, "autocontrast", 3),
        sub_policy(0.9, "translateY", 9, 0.7, "translateY", 9),
        sub_policy(0.9, "autocontrast", 2, 0.8, "solarize", 3),
        sub_policy(0.8, "equalize", 8, 0.1, "invert", 3),
        sub_policy(0.7, "translateY", 9, 0.9, "autocontrast", 1),
    ]

    return policies


def autoaugment(image, augmentation_name):
    available_policies = {'CIFAR10': cifar10_policy}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))
    policies = available_policies[augmentation_name]()
    image = select_and_apply_random_policy(policies, image)
    return image
