import tensorflow as tf


def smart_constant_value(pred):
    """Return the bool value for `pred`, or None if `pred` had a dynamic value.

    Arguments:
      pred: A scalar, either a Python bool or tensor.

    Returns:
      True or False if `pred` has a constant boolean value, None otherwise.

    Raises:
      TypeError: If `pred` is not a Tensor or bool.
    """
    if isinstance(pred, tf.Tensor):
        pred_value = tf.constant(pred)
    elif pred in {0, 1}:  # Accept 1/0 as valid boolean values
        pred_value = bool(pred)
    elif isinstance(pred, bool):
        pred_value = pred
    else:
        raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                        "Found instead: %s" % type(pred))

    return pred_value


def _smart_cond(pred, true_fn=None, false_fn=None, name=None):
    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

    If `pred` is a bool or has a constant value, we return either `true_fn()`
    or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

    Arguments:
      pred: A scalar determining whether to return the result of `true_fn` or
        `false_fn`.
      true_fn: The callable to be performed if pred is true.
      false_fn: The callable to be performed if pred is false.
      name: Optional name prefix when using `tf.cond`.

    Returns:
      Tensors returned by the call to either `true_fn` or `false_fn`.

    Raises:
      TypeError: If `true_fn` or `false_fn` is not callable.
    """
    if not callable(true_fn):
        raise TypeError("`true_fn` must be callable.")
    if not callable(false_fn):
        raise TypeError("`false_fn` must be callable.")

    pred_value = smart_constant_value(pred)
    if pred_value is not None:
        if pred_value:
            return true_fn()
        else:
            return false_fn()
    else:
        return tf.cond(pred, true_fn=true_fn, false_fn=false_fn,
                       name=name)


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

    If `pred` is a bool or has a constant value, we return either `true_fn()`
    or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

    Arguments:
      pred: A scalar determining whether to return the result of `true_fn` or
        `false_fn`.
      true_fn: The callable to be performed if pred is true.
      false_fn: The callable to be performed if pred is false.
      name: Optional name prefix when using `tf.cond`.

    Returns:
      Tensors returned by the call to either `true_fn` or `false_fn`.

    Raises:
      TypeError: If `true_fn` or `false_fn` is not callable.
    """
    if isinstance(pred, tf.Variable):
        return tf.cond(
            pred, true_fn=true_fn, false_fn=false_fn, name=name)

    return _smart_cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
