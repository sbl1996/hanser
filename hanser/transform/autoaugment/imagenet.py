import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, solarize_add, autocontrast, translate_x, \
    translate_y, rotate, color, posterize, contrast, brightness, equalize, invert, cutout2 as cutout, random_apply

# This implementation was stolen from https://github.com/google/automl/blob/master/efficientnetv2/autoaugment.py
# 1. Image ops are exactly the same.
# 2. The function to combine and apply policies is different.


_MAX_LEVEL = 10.
_FILL_COLOR = (128, 128, 128)


CONSTS = {
    'cutout_const': 40,
    'translate_const': 100,
}


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level):
    level = level / _MAX_LEVEL * 30.
    level = _randomly_negate_tensor(level)
    return level


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return 1.0
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return level


def _enhance_level_to_arg(level):
    return (level / _MAX_LEVEL) * 1.8 + 0.1


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * float(CONSTS['translate_const'])
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level):
    level = int((level / _MAX_LEVEL) * 4)
    return level


def _solarize_level_to_arg(level):
    level = int((level / _MAX_LEVEL) * 256)
    return level


def _solarize_add_level_to_arg(level):
    level = int((level / _MAX_LEVEL) * 110)
    return level


def _cutout_level_to_arg(level):
    return int((level / _MAX_LEVEL) * CONSTS['cutout_const'])


def _shear_x(img, level):
    return shear_x(img, _shear_level_to_arg(level), _FILL_COLOR)


def _shear_y(img, level):
    return shear_y(img, _shear_level_to_arg(level), _FILL_COLOR)


def _translate_x(img, level):
    return translate_x(img, _translate_level_to_arg(level), _FILL_COLOR)


def _translate_y(img, level):
    return translate_y(img, _translate_level_to_arg(level), _FILL_COLOR)


def _rotate(img, level):
    return rotate(img, _rotate_level_to_arg(level), _FILL_COLOR)


def _color(img, level):
    return color(img, _enhance_level_to_arg(level))


def _posterize(img, level):
    return posterize(img, _posterize_level_to_arg(level))


def _solarize(img, level):
    return solarize(img, _solarize_level_to_arg(level))


def _solarize_add(img, level):
    return solarize_add(img, _solarize_add_level_to_arg(level))


def _contrast(img, level):
    return contrast(img, _enhance_level_to_arg(level))


def _sharpness(img, level):
    return sharpness(img, _enhance_level_to_arg(level))


def _brightness(img, level):
    return brightness(img, _enhance_level_to_arg(level))


# noinspection PyUnusedLocal
def _autocontrast(img, level):
    return autocontrast(img)


# noinspection PyUnusedLocal
def _equalize(img, level):
    return equalize(img)


# noinspection PyUnusedLocal
def _invert(img, level):
    return invert(img)


def _cutout(img, level):
    return cutout(img, _cutout_level_to_arg(level), _FILL_COLOR)


NAME_TO_FUNC = {
    "shearX": _shear_x,
    "shearY": _shear_y,
    'translateX': _translate_x,
    'translateY': _translate_y,
    "rotate": _rotate,
    "color": _color,
    "posterize": _posterize,
    "solarize": _solarize,
    "solarize_add": _solarize_add,
    "contrast": _contrast,
    "sharpness": _sharpness,
    "brightness": _brightness,
    "autocontrast": _autocontrast,
    "equalize": _equalize,
    "invert": _invert,
    'cutout': _cutout,
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


def imagenet_policy_v0():
    policies = [
        sub_policy(0.8, 'equalize', 1, 0.8, 'shearY',   4),
        sub_policy(0.4, 'color',    9, 0.6, 'equalize', 3),
        sub_policy(0.4, 'color',    1, 0.6, 'rotate',   8),
        sub_policy(0.8, 'solarize', 3, 0.4, 'equalize', 7),
        sub_policy(0.4, 'solarize', 2, 0.6, 'solarize', 2),

        sub_policy(0.2, 'color',    0, 0.8, 'equalize',     8),
        sub_policy(0.4, 'equalize', 8, 0.8, 'solarize_add', 3),
        sub_policy(0.2, 'shearX',   9, 0.6, 'rotate',       8),
        sub_policy(0.6, 'color',    1, 1.0, 'equalize',     2),
        sub_policy(0.4, 'invert',   9, 0.6, 'rotate',       0),

        sub_policy(1.0, 'equalize',  9, 0.6, 'shearY',       3),
        sub_policy(0.4, 'color',     7, 0.6, 'equalize',     0),
        sub_policy(0.4, 'posterize', 6, 0.4, 'autocontrast', 7),
        sub_policy(0.6, 'solarize',  8, 0.6, 'color',        9),
        sub_policy(0.2, 'solarize',  4, 0.8, 'rotate',       9),

        sub_policy(1.0, 'rotate',   7, 0.8, 'translateY', 9),
        sub_policy(0.0, 'shearX',   0, 0.8, 'solarize',   4),
        sub_policy(0.8, 'shearY',   0, 0.6, 'color',      4),
        sub_policy(1.0, 'color',    0, 0.6, 'rotate',     2),
        sub_policy(0.8, 'equalize', 4, 0.0, 'equalize',   8),

        sub_policy(1.0, 'equalize',  4, 0.6, 'autocontrast', 2),
        sub_policy(0.4, 'shearY',    7, 0.6, 'solarize_add', 7),
        sub_policy(0.8, 'posterize', 2, 0.6, 'solarize',     10),
        sub_policy(0.6, 'solarize',  8, 0.6, 'equalize',     1),
        sub_policy(0.8, 'color',     6, 0.4, 'rotate',       5),
    ]
    # policies = [
    #     sub_policy(1.0, 'cutout', 10, 1.0, 'translateY', 3),
    # ]
    return policies


def autoaugment(image):
    CONSTS['cutout_const'] = 100
    CONSTS['translate_const'] = 250

    policies = imagenet_policy_v0()
    image = select_and_apply_random_policy(policies, image)
    return image


def randaugment(image, num_layers=2, magnitude=10):
    CONSTS['cutout_const'] = 40
    CONSTS['translate_const'] = 100

    available_ops = [
        'autocontrast', 'equalize', 'invert', 'rotate', 'posterize',
        'solarize', 'color', 'contrast', 'brightness', 'sharpness',
        'shearX', 'shearY', 'translateX', 'translateY', 'cutout', 'solarize_add',
    ]
    # available_ops = [
    #     'cutout', 'translateY',
    # ]

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform(
            [], maxval=len(available_ops), dtype=tf.int32)
        random_magnitude = float(magnitude)
        for (i, op_name) in enumerate(available_ops):
            prob = tf.random.uniform((), minval=0.2, maxval=0.8, dtype=tf.float32)
            selected_func = lambda im: NAME_TO_FUNC[op_name](im, random_magnitude)
            image = tf.cond(
                tf.equal(i, op_to_select),
                lambda: random_apply(selected_func, prob, image),
                lambda: image)
    return image


def rand_or_auto_augment(image, num_layers=2, magnitude=10):
    i = tf.random.uniform((), maxval=2, dtype=tf.int32)
    image = tf.cond(
        tf.equal(i, 0), lambda: autoaugment(image),
        lambda: image)
    image = tf.cond(
        tf.equal(i, 1), lambda: randaugment(image, num_layers, magnitude),
        lambda: image)
    return image
