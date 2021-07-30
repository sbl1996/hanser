import tensorflow as tf

def _is_batch(x):
    return len(x.shape) == 4

def _wrap_batch(tensors, is_batch):
    return tuple(t if is_batch else t[None] for t in tensors)


def _unwrap_batch(tensors, is_batch):
    return tuple(t if is_batch else t[0] for t in tensors)


def image_dimensions(image, rank):
    """Returns the dimensions of an image tensor.
    Args:
        image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
        rank: The expected rank of the image
    Returns:
        A list of corresponding to the dimensions of the input image. Dimensions
        that are statically known are python integers, otherwise they are integer
        scalar tensors.
    """
    if image.shape.is_fully_defined():
        return image.shape.as_list()
    else:
        static_shape = image.shape.with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [
            s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
        ]


def get_ndims(image):
    return image.shape.ndims or tf.rank(image)


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D tensor.
    Returns:
      4D tensor with the same type.
    """
    with tf.control_dependencies(
        [
            tf.debugging.assert_rank_in(
                image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
            )
        ]
    ):
        ndims = image.shape.ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.
    Args:
      image: 4D tensor.
      ndims: The original rank of the image.
    Returns:
      `ndims`-D tensor with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image