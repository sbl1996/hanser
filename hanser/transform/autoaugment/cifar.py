from hanser.transform.autoaugment.common import apply_autoaugment
from hanser.transform.autoaugment.imagenet import randaugment, trival_augment

def cifar10_policy():
    policies = [
        (0.1, "invert",       7, 0.2, "contrast",     6),
        (0.7, "rotate",       2, 0.3, "translateX",   9),
        (0.8, "sharpness",    1, 0.9, "sharpness",    3),
        (0.5, "shearY",       8, 0.7, "translateY",   9),
        (0.5, "autocontrast", 8, 0.9, "equalize",     2),

        (0.2, "shearY",       7, 0.3, "posterize",    7),
        (0.4, "color",        3, 0.6, "brightness",   7),
        (0.3, "sharpness",    9, 0.7, "brightness",   9),
        (0.6, "equalize",     5, 0.5, "equalize",     1),
        (0.6, "contrast",     7, 0.6, "sharpness",    5),
        #
        (0.7, "color",        7, 0.5, "translateX",   8),
        (0.3, "equalize",     7, 0.4, "autocontrast", 8),
        (0.4, "translateY",   3, 0.2, "sharpness",    6),
        (0.9, "brightness",   6, 0.2, "color",        8),
        (0.5, "solarize",     2, 0.0, "invert",       3),

        (0.2, "equalize",     0, 0.6, "autocontrast", 0),
        (0.2, "equalize",     8, 0.8, "equalize",     4),
        (0.9, "color",        9, 0.6, "equalize",     6),
        (0.8, "autocontrast", 4, 0.2, "solarize",     8),
        (0.1, "brightness",   3, 0.7, "color",        0),

        (0.4, "solarize",     5, 0.9, "autocontrast", 3),
        (0.9, "translateY",   9, 0.7, "translateY",   9),
        (0.9, "autocontrast", 2, 0.8, "solarize",     3),
        (0.8, "equalize",     8, 0.1, "invert",       3),
        (0.7, "translateY",   9, 0.9, "autocontrast", 1),
    ]
    return policies


def autoaugment(image, augmentation_name='CIFAR10'):
    available_policies = {'CIFAR10': cifar10_policy}
    policies = available_policies[augmentation_name]()

    hparams = {
        "translate_max": 10 / 32, # https://github.com/tensorflow/tpu/issues/637
    }

    return apply_autoaugment(image, policies, hparams)