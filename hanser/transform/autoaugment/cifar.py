import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, autocontrast, translate_x, \
    translate_y, rotate, color, posterize, contrast, brightness, equalize, invert

_MAX_LEVEL = 9
_IMG_SIZE = 32
_FILL_COLOR = (128, 128, 128)


def _randomly_negate_tensor(tensor):
    return tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: tensor,
        lambda: -tensor
    )


def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level


def _enhance_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.9
    level = _randomly_negate_tensor(level)
    return 1 + level


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * (_IMG_SIZE * 150 / 331)
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level):
    level = (8, 8, 7, 7, 6, 6, 5, 5, 4, 4)[level]
    return level


def _solarize_level_to_arg(level):
    level = (_MAX_LEVEL - level) / _MAX_LEVEL * 256
    return level


def _shear_x(img, level):
    return shear_x(img, _shear_level_to_arg(level), _FILL_COLOR)


def _shear_y(img, level):
    return shear_x(img, _shear_level_to_arg(level), _FILL_COLOR)


def _op_shear_x(img, level):
    return shear_x(img, _shear_level_to_arg(level), _FILL_COLOR)


def _op_shear_y(img, level):
    return shear_y(img, _shear_level_to_arg(level), _FILL_COLOR)


def _op_translate_x(img, level):
    return translate_x(img, _translate_level_to_arg(level), _FILL_COLOR)


def _op_translate_y(img, level):
    return translate_y(img, _translate_level_to_arg(level), _FILL_COLOR)


def _op_rotate(img, level):
    return rotate(img, _rotate_level_to_arg(level), _FILL_COLOR)


def _op_color(img, level):
    return color(img, _enhance_level_to_arg(level))


def _op_posterize(img, level):
    return posterize(img, _posterize_level_to_arg(level))


def _op_solarize(img, level):
    return solarize(img, _solarize_level_to_arg(level))


def _op_contrast(img, level):
    return contrast(img, _enhance_level_to_arg(level))


def _op_sharpness(img, level):
    return sharpness(img, _enhance_level_to_arg(level))


def _op_brightness(img, level):
    return brightness(img, _enhance_level_to_arg(level))


def _op_autocontrast(img, level):
    return autocontrast(img)


def _op_equalize(img, level):
    return equalize(img)


def _op_invert(img, level):
    return invert(img)


NAME_TO_FUNC = {
    "shearX": _op_shear_x,
    "shearY": _op_shear_y,
    'translateX': _op_translate_x,
    'translateY': _op_translate_y,
    "rotate": _op_rotate,
    "color": _op_color,
    "posterize": _op_posterize,
    "solarize": _op_solarize,
    "contrast": _op_contrast,
    "sharpness": _op_sharpness,
    "brightness": _op_brightness,
    "autocontrast": _op_autocontrast,
    "equalize": _op_equalize,
    "invert": _op_invert,
}


def _apply_func_with_prob(func, p, image, level):
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + p), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, level),
        lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""

    policy_to_select = tf.random.uniform((), maxval=len(policies), dtype=tf.int32)
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
        #
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
