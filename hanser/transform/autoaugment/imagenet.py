# We decided to follow the original paper rather than official code
# especially for translate_const and cutout_const

import tensorflow as tf

from hanser.transform.autoaugment.common import apply_autoaugment, NAME_TO_FUNC, H_PARAMS, random_apply


def imagenet_policy_v0():
    policies = [
        (0.8, 'equalize',  1, 0.8, 'shearY',       4),
        (0.4, 'color',     9, 0.6, 'equalize',     3),
        (0.4, 'color',     1, 0.6, 'rotate',       8),
        (0.8, 'solarize',  3, 0.4, 'equalize',     7),
        (0.4, 'solarize',  2, 0.6, 'solarize',     2),

        (0.2, 'color',     0, 0.8, 'equalize',     8),
        (0.4, 'equalize',  8, 0.8, 'solarize_add', 3),
        (0.2, 'shearX',    9, 0.6, 'rotate',       8),
        (0.6, 'color',     1, 1.0, 'equalize',     2),
        (0.4, 'invert',    9, 0.6, 'rotate',       0),

        (1.0, 'equalize',  9, 0.6, 'shearY',       3),
        (0.4, 'color',     7, 0.6, 'equalize',     0),
        (0.4, 'posterize', 6, 0.4, 'autocontrast', 7),
        (0.6, 'solarize',  8, 0.6, 'color',        9),
        (0.2, 'solarize',  4, 0.8, 'rotate',       9),

        (1.0, 'rotate',    7, 0.8, 'translateY',   9),
        (0.0, 'shearX',    0, 0.8, 'solarize',     4),
        (0.8, 'shearY',    0, 0.6, 'color',        4),
        (1.0, 'color',     0, 0.6, 'rotate',       2),
        (0.8, 'equalize',  4, 0.0, 'equalize',     8),

        (1.0, 'equalize',  4, 0.6, 'autocontrast', 2),
        (0.4, 'shearY',    7, 0.6, 'solarize_add', 7),
        (0.8, 'posterize', 2, 0.6, 'solarize',     10),
        (0.6, 'solarize',  8, 0.6, 'equalize',     1),
        (0.8, 'color',     6, 0.4, 'rotate',       5),
    ]
    return policies


def autoaugment(image):
    # hparams = {
        # 'cutout_max': 100, # No cutout in autoaugment
        # 'translate_max': 250, # Use 150 / 331, follow paper rather than code
    # }
    policies = imagenet_policy_v0()
    return apply_autoaugment(image, policies)


def get_augmentation_space(name):
    if isinstance(name, list):
        return name
    if name in ['aa', 'ua']:
        return [
            'identity', 'autocontrast', 'equalize', 'rotate',
            'solarize', 'color', 'posterize', 'contrast', 'brightness',
            'sharpness', 'shearX', 'shearY', 'translateX', 'translateY',
            'invert', 'cutout',
            # 'cutout', 'invert', 'samplePairing'
            # cutout actually not appeared, samplePairing not implemented
            # UA = AA − { SamplePairing }
        ]
    elif name in ['ra']:
        return [
            'identity', 'autocontrast', 'equalize', 'rotate',
            'solarize', 'color', 'posterize', 'contrast', 'brightness',
            'sharpness', 'shearX', 'shearY', 'translateX', 'translateY',
            # RA = AA − { SamplePairing, Invert, Cutout }
        ]
    elif name in ['aa-']:
        return [
            'identity', 'autocontrast', 'equalize', 'rotate',
            'solarize', 'color', 'posterize', 'contrast', 'brightness',
            'sharpness', 'shearX', 'shearY', 'translateX', 'translateY',
            'cutout',
            # remove the extreme invert operation
            # performs very well for CIFAR-10, but not great for SVHN Core
            # Conclusion from TrivialAugment 4.2.1
        ]
    elif name in ['ra0']: # official
        return [
            'autocontrast', 'equalize', 'rotate', 'solarize', 'color',
            'posterize', 'contrast', 'brightness', 'sharpness',
            'shearX', 'shearY', 'translateX', 'translateY',
            'cutout', 'solarize_add', 'invert',
        ]
    elif name in ['ra+']:
        return [
            'autocontrast', 'equalize', 'rotate', 'solarize', 'color',
            'contrast', 'brightness', 'sharpness',
            'shearX', 'shearY', 'translateX', 'translateY',
            'cutout', 'solarize_add',
            # RA0 - { Invert, Posterize }
        ]
    else:
        raise ValueError("No augmentation space: %s" % name)

# official
def randaugment(image, num_layers=2, magnitude=10.,
                random_magnitude=False, augmentation_space='ra0'):

    # RandAugment use larger translate or cutout for EfficientNet
    # with higher image resolution, therefore the relative ratio of
    # translate or cutout to image resolution is almost constant.
    # We divide ratio to estimate this behavieor.
    # Official code:
    # cutout_const: 40, translate_const: 100
    # resolution|magnitude: 224|9, 456|17, 600|28
    ratio = (magnitude / H_PARAMS['max_level'])
    hparams = {
        **H_PARAMS,
        "translate_max": 150 / 331 / ratio,
        "cutout_max": 60 / 331 / ratio,
    }

    available_ops = get_augmentation_space(augmentation_space)

    op_funcs = [
        NAME_TO_FUNC[op_name] for op_name in available_ops
    ]

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform((), 0, maxval=len(available_ops), dtype=tf.int32)
        for (i, op_func) in enumerate(op_funcs):
            prob = tf.random.uniform((), minval=0.2, maxval=0.8, dtype=tf.float32)
            if random_magnitude:
                selected_func = lambda im: op_func(im, random_level(magnitude), hparams)
            else:
                selected_func = lambda im: op_func(im, tf.cast(magnitude, tf.float32), hparams)
            image = tf.cond(
                tf.equal(i, op_to_select),
                lambda: random_apply(selected_func, prob, image),
                lambda: image)
    return image


# timm
def randaugment_t(image, num_layers=2, magnitude=10.):
    magnitude = tf.cast(magnitude, tf.float32)

    hparams = {**H_PARAMS}

    available_ops = [
        'autocontrast', 'equalize', 'invert', 'rotate', 'posterize',
        'solarize', 'color', 'contrast', 'brightness', 'sharpness',
        'shearX', 'shearY', 'translateX', 'translateY', 'solarize_add',
        # Use RE-M afterwards, instead of cutout here
    ]

    op_funcs = [
        NAME_TO_FUNC[op_name] for op_name in available_ops
    ]

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform((), 0, maxval=len(available_ops), dtype=tf.int32)
        for (i, op_func) in enumerate(op_funcs):
            selected_func = lambda im: op_func(im, magnitude, hparams)
            image = tf.cond(
                tf.equal(i, op_to_select),
                lambda: random_apply(selected_func, 0.5, image),
                lambda: image)
    return image


def rand_or_auto_augment(image, num_layers=2, magnitude=10.):
    i = tf.random.uniform((), maxval=2, dtype=tf.int32)
    image = tf.cond(
        tf.equal(i, 0), lambda: autoaugment(image),
        lambda: image)
    image = tf.cond(
        tf.equal(i, 1), lambda: randaugment(image, num_layers, magnitude),
        lambda: image)
    return image


def random_level(max_level):
    max_level = tf.cast(max_level, tf.int32)
    level = tf.random.uniform((), 0, max_level, dtype=tf.int32)
    return tf.cast(level, tf.float32)


def trivial_augment(image):
    hparams = {
        **H_PARAMS,
        'max_level': 30,
    }

    available_ops = [
        'identity' ,'autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
        'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY',
    ]

    op_funcs = [
        NAME_TO_FUNC[op_name] for op_name in available_ops
    ]

    op_to_select = tf.random.uniform((), 0, maxval=len(available_ops), dtype=tf.int32)
    for i, op_func in enumerate(op_funcs):
        image = tf.cond(
            tf.equal(i, op_to_select),
            lambda: op_func(image, random_level(hparams['max_level']), hparams),
            lambda: image)
    return image