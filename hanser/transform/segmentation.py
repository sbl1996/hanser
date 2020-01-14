import tensorflow as tf


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random.uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.linspace(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random.shuffle(scale_factors)
    return shuffled_scale_factors[0]


def random_scale(image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [height, width, 1].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(image)
    new_dim = tf.cast(
        tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale,
        tf.int32)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.image.resize(image, new_dim, method='bilinear')
    if label is not None:
        label = tf.image.resize(label, new_dim, method='nearest')

    return image, label


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

        image, depths, normals = random_crop([image, depths, normals], 120, 150)

    Args:
        image_list: a list of image tensors of the same dimension but possibly
            varying channel.
        crop_height: the new height.
        crop_width: the new width.

    Returns:
        the image_list with cropped images.

    Raises:
        ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor [expected] [actual]', 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor [expected][actual]', height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor [expected][actual]', width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random.uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random.uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random.uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
        image: an image of shape [height, width, channels].
        offset_height: a scalar tensor indicating the height offset.
        offset_width: a scalar tensor indicating the width offset.
        crop_height: the height of the cropped image.
        crop_width: the width of the cropped image.

    Returns:
        The cropped (and resized) image.

    Raises:
        ValueError: if `image` doesn't have rank of 3.
        InvalidArgumentError: if the rank is not 3 or if the image dimensions are
            less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def flip_dim(tensor_list, prob=0.5, dim=1):
    """Randomly flips a dimension of the given tensor.

    The decision to randomly flip the `Tensors` is made together. In other words,
    all or none of the images pass in are flipped.

    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
    that we can control for the probability as well as ensure the same decision
    is applied across the images.

    Args:
        tensor_list: A list of `Tensors` with the same number of dimensions.
        prob: The probability of a left-right flip.
        dim: The dimension to flip, 0, 1, ..

    Returns:
        outputs: A list of the possibly flipped `Tensors` as well as an indicator
        `Tensor` at the end whose value is `True` if the inputs were flipped and
        `False` otherwise.

    Raises:
        ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random.uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    # outputs.append(is_flipped)

    return outputs


def rot90(tensor_list, prob=0.5, k=1):
    """Randomly flips a dimension of the given tensor.

    The decision to randomly flip the `Tensors` is made together. In other words,
    all or none of the images pass in are flipped.

    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
    that we can control for the probability as well as ensure the same decision
    is applied across the images.

    Args:
        tensor_list: A list of `Tensors` with the same number of dimensions.
        prob: The probability of a left-right flip.
        k: A scalar integer. The number of times the image is rotated by 90 degrees.

    Returns:
        outputs: A list of the possibly flipped `Tensors` as well as an indicator
        `Tensor` at the end whose value is `True` if the inputs were flipped and
        `False` otherwise.

    Raises:
        ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random.uniform([])

    def rotate():
        rotated = []
        for tensor in tensor_list:
            rotated.append(tf.image.rot90(tensor, k=k))
        return rotated

    is_rotated = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_rotated, rotate, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    return outputs

