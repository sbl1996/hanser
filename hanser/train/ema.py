import tensorflow as tf


def get_ema_vars(model):
    ema_vars = model.trainable_variables + [
        v for v in model.variables
        if 'moving_mean' in v.name or 'moving_variance' in v.name
    ]
    return ema_vars


def swap_weights(avg_vars, model_vars):
    """Swap the average and moving weights.

    This is a convenience method to allow one to evaluate the averaged weights
    at test time. Loads the weights stored in `self._average_weights` into the model,
    keeping a copy of the original model weights. Swapping twice will return
    the original weights.
    """
    if tf.distribute.in_cross_replica_context():
        strategy = tf.distribute.get_strategy()
        strategy.run(_swap_weights_dist, args=(avg_vars, model_vars))
    else:
        _swap_weights_local(avg_vars, model_vars)


@tf.function
def _swap_weights_local(avg_vars, model_vars):
    for a, b in zip(avg_vars, model_vars):
        a.assign_add(b)
        b.assign(a - b)
        a.assign_sub(b)


@tf.function
def _swap_weights_dist(avg_vars, model_vars):
    def fn_0(a, b):
        return a.assign_add(b)

    def fn_1(b, a):
        return b.assign(a - b)

    def fn_2(a, b):
        return a.assign_sub(b)

    def swap(strategy, a, b):
        """Swap `a` and `b` and mirror to all devices."""
        for a_element, b_element in zip(a, b):
            strategy.extended.update(
                a_element, fn_0, args=(b_element,)
            )  # a = a + b
            strategy.extended.update(
                b_element, fn_1, args=(a_element,)
            )  # b = a - b
            strategy.extended.update(
                a_element, fn_2, args=(b_element,)
            )  # a = a - b

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(swap, args=(avg_vars, model_vars))
