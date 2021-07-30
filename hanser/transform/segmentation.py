import tensorflow as tf
from hanser.transform import image_dimensions


def get_random_scale(min_scale_factor, max_scale_factor, step_size):

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

    if step_size == 0:
        return tf.random.uniform([1], minval=min_scale_factor, maxval=max_scale_factor)

    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.linspace(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random.shuffle(scale_factors)
    return shuffled_scale_factors[0]


def random_scale(image, label, scale_factor_range=(0.5, 2.0), step_size=0.25):
    scale = get_random_scale(scale_factor_range[0], scale_factor_range[1], step_size)
    image_shape = tf.shape(image)
    new_dim = tf.cast(
        tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale, tf.int32)

    image = tf.image.resize(image, new_dim, method='bilinear')
    label = tf.image.resize(label, new_dim, method='nearest')

    return image, label


def random_crop(image_list, size):
    crop_height, crop_width = size
    height, width, c = image_dimensions(image_list[0], 3)
    max_offset_height = height - crop_height + 1
    max_offset_width = width - crop_width + 1
    offset_height = tf.random.uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random.uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def pad(image, label, size, image_pad_value, label_pad_value):
    """
    Pad image and label to *at least* `size`.

    Args:
        image (tf.Tensor): 3-D Tensor of shape `[height, width, channels]`.
        label (tf.Tensor): 3-D Tensor of shape `[height, width, channels]`
            with the same size as `image`.
        size (tuple): Padded size.
        image_pad_value: Pad values to use for `image`.
        label_pad_value: Scalar pad values to use for `label`.
    """
    height, width, depth = image_dimensions(image, rank=3)

    offset_height, offset_width = 0, 0
    target_height = tf.maximum(size[0], height)
    target_width = tf.maximum(size[1], width)
    ph1, ph2 = offset_height, target_height - offset_height - height
    pw1, pw2 = offset_width, target_width - offset_width - width

    image -= image_pad_value

    image = tf.pad(image, [(ph1, ph2), (pw1, pw2), (0, 0)])
    label = tf.pad(label, [(ph1, ph2), (pw1, pw2), (0, 0)],
                   constant_values=tf.cast(label_pad_value, label.dtype))
    image = image + image_pad_value

    return image, label


def flip_dim(tensor_list, prob=0.5, dim=1):
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

    return outputs


def rot90(tensor_list, prob=0.5, k=1):

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