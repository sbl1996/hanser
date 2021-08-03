import tensorflow as tf


def evonorm(inputs, is_training, nonlinearity=True, init_zero=False, decay=0.9, epsilon=1e-5):

    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    var_shape = (1, 1, 1, inputs.shape[3])
    with tf.variable_scope(None, default_name='evonorm'):
        beta = tf.get_variable(
            'beta',
            shape=var_shape,
            dtype=inputs.dtype,
            initializer=tf.zeros_initializer())
        gamma = tf.get_variable(
            'gamma',
            shape=var_shape,
            dtype=inputs.dtype,
            initializer=gamma_initializer)
        if nonlinearity:
            v = tf.get_variable(
                'v',
                shape=var_shape,
                dtype=inputs.dtype,
                initializer=tf.ones_initializer())
            left = _batch_std(
                inputs,
                decay=decay,
                epsilon=epsilon,
                training=is_training)
            right = v * inputs + _instance_std(inputs, epsilon=epsilon)
            inputs = inputs / tf.maximum(left, right)
    return inputs * gamma + beta


def _batch_std(inputs, training, decay=0.9, epsilon=1e-5):
    """Batch standard deviation."""
    var_shape, axes = (1, 1, 1, inputs.shape[3]), [0, 1, 2]
    moving_variance = tf.get_variable(
        name="moving_variance",
        shape=var_shape,
        initializer=tf.initializers.ones(),
        dtype=tf.float32,
        collections=[
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES
        ],
        trainable=False)
    if training:
        _, variance = tf.nn.moments(inputs, axes, keep_dims=True)
        variance = tf.cast(variance, tf.float32)
        update_op = tf.assign_sub(
            moving_variance,
            (moving_variance - variance) * (1 - decay))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
    else:
        variance = moving_variance
    std = tf.sqrt(variance + epsilon)
    return tf.cast(std, inputs.dtype)


def _instance_std(inputs, epsilon=1e-5):
    """Instance standard deviation."""
    axes = [1, 2]
    _, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
    return tf.sqrt(variance + epsilon)
