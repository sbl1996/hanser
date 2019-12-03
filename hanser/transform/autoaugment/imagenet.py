import tensorflow as tf
from hanser.transform import sharpness, shear_x, shear_y, solarize, solarize_add, autocontrast, translate_x, \
    translate_y, rotate, \
    color, posterize, contrast, brightness, equalize, invert

CONSTS = {
    'max_level': 10,
    'fill_color': (128, 128, 128),
    'cutout_const': 40,
    'translate_const': 100,
}


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level):
    level = (level / CONSTS['max_level']) * 30.
    level = _randomly_negate_tensor(level)
    return level


def _enhance_level_to_arg(level):
    return (level / CONSTS['max_level']) * 1.8 + 0.1


def _shear_level_to_arg(level):
    level = (level / CONSTS['max_level']) * 0.3
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level):
    level = (level / CONSTS['max_level']) * float(CONSTS['translate_const'])
    level = _randomly_negate_tensor(level)
    return level


def _posterize_level_to_arg(level):
    level = int((level / CONSTS['max_level']) * 4)
    return level


def _solarize_level_to_arg(level):
    level = int((level / CONSTS['max_level']) * 256)
    return level


def _solarize_add_level_to_arg(level):
    level = int((level / CONSTS['max_level']) * 110)
    return level


def _cutout_level_to_arg(level):
    return int((level / CONSTS['max_level']) * CONSTS['cutout_const'])


def cutout(image, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.
  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
  Returns:
    An image Tensor that is of type uint8.
  """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height,
        dtype=tf.int32)

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width,
        dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace,
        image)
    return image


def _shear_x(img, level):
    return shear_x(img, _shear_level_to_arg(level), CONSTS['fill_color'])


def _shear_y(img, level):
    return shear_y(img, _shear_level_to_arg(level), CONSTS['fill_color'])


def _translate_x(img, level):
    return translate_x(img, _translate_level_to_arg(level), CONSTS['fill_color'])


def _translate_y(img, level):
    return translate_y(img, _translate_level_to_arg(level), CONSTS['fill_color'])


def _rotate(img, level):
    return rotate(img, _rotate_level_to_arg(level), CONSTS['fill_color'])


def _color(img, level):
    return color(img, _enhance_level_to_arg(level))


def _posterize(img, level):
    return posterize(img, _posterize_level_to_arg(level))


def _solarize(img, level):
    return solarize(img, _solarize_level_to_arg(level))


def _solarize_add(img, level):
    return solarize_add(img, _solarize_level_to_arg(level))


def _contrast(img, level):
    return contrast(img, _enhance_level_to_arg(level))


def _sharpness(img, level):
    return sharpness(img, _enhance_level_to_arg(level))


def _brightness(img, level):
    return brightness(img, _enhance_level_to_arg(level))


def _autocontrast(img, level):
    return autocontrast(img)


def _equalize(img, level):
    return equalize(img)


def _invert(img, level):
    return invert(img)


def _cutout(img, level):
    return cutout(img, _cutout_level_to_arg(level), CONSTS['fill_color'])


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
    """autoaugment policy that was used in autoaugment paper."""
    # each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policies = [
        sub_policy('equalize', 0.8, 1, 'sheary', 0.8, 4),
        sub_policy('color', 0.4, 9, 'equalize', 0.6, 3),
        sub_policy('color', 0.4, 1, 'rotate', 0.6, 8),
        sub_policy('solarize', 0.8, 3, 'equalize', 0.4, 7),
        sub_policy('solarize', 0.4, 2, 'solarize', 0.6, 2),

        sub_policy('color', 0.2, 0, 'equalize', 0.8, 8),
        sub_policy('equalize', 0.4, 8, 'solarize_add', 0.8, 3),
        sub_policy('shearX', 0.2, 9, 'rotate', 0.6, 8),
        sub_policy('color', 0.6, 1, 'equalize', 1.0, 2),
        sub_policy('invert', 0.4, 9, 'rotate', 0.6, 0),

        sub_policy('equalize', 1.0, 9, 'shearY', 0.6, 3),
        sub_policy('color', 0.4, 7, 'equalize', 0.6, 0),
        sub_policy('posterize', 0.4, 6, 'autocontrast', 0.4, 7),
        sub_policy('solarize', 0.6, 8, 'color', 0.6, 9),
        sub_policy('solarize', 0.2, 4, 'rotate', 0.8, 9),

        sub_policy('rotate', 1.0, 7, 'translateY', 0.8, 9),
        sub_policy('shearX', 0.0, 0, 'solarize', 0.8, 4),
        sub_policy('shearY', 0.8, 0, 'color', 0.6, 4),
        sub_policy('color', 1.0, 0, 'rotate', 0.6, 2),
        sub_policy('equalize', 0.8, 4, 'equalize', 0.0, 8),

        sub_policy('equalize', 1.0, 4, 'autocontrast', 0.6, 2),
        sub_policy('shearY', 0.4, 7, 'solarize_add', 0.6, 7),
        sub_policy('posterize', 0.8, 2, 'solarize', 0.6, 10),
        sub_policy('solarize', 0.6, 8, 'equalize', 0.6, 1),
        sub_policy('color', 0.8, 6, 'rotate', 0.4, 5),
    ]
    return policies


def autoaugment(image, augmentation_name):
    CONSTS['cutout_const'] = 100
    CONSTS['translate_const'] = 250

    available_policies = {'ImageNet': imagenet_policy_v0}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))
    policies = available_policies[augmentation_name]()
    image = select_and_apply_random_policy(policies, image)
    return image


def rand_augment(image, num_layers, magnitude):
    CONSTS['cutout_const'] = 40
    CONSTS['translate_const'] = 100

    available_ops = [
        'autocontrast', 'equalize', 'invert', 'solarize_add', 'posterize',
        'solarize', 'color', 'contrast', 'brightness', 'sharpness', 'cutout',
        'shearX', 'shearY', 'translateX', 'translateY', 'rotate',
    ]

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform(
            [], maxval=len(available_ops), dtype=tf.int32)
        random_magnitude = float(magnitude)
        for (i, op_name) in enumerate(available_ops):
            selected_func = NAME_TO_FUNC[op_name]
            image = tf.cond(
                tf.equal(i, op_to_select),
                lambda: selected_func(image, random_magnitude),
                lambda: image)
    return image
