import tensorflow as tf


def get_ema_vars(model):
    ema_vars = model.trainable_variables + [
        v for v in model.variables
        if 'moving_mean' in v.name or 'moving_variance' in v.name
    ]
    return ema_vars