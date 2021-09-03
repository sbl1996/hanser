import tensorflow as tf

def log_uniform(shape, minval, maxval, dtype=tf.float32):
    minval = tf.math.log(minval)
    maxval = tf.math.log(maxval)
    x = tf.random.uniform(shape, minval, maxval, dtype)
    return tf.exp(x)


@tf.function
def random_erasing(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, max_random_value=1.0):

    if tf.random.uniform(()) > p:
        return image

    shape = tf.shape(image)
    height, width, c = shape[0], shape[1], shape[2]
    area = tf.cast(height * width, tf.float32)

    while tf.constant(True):
        target_area = tf.random.uniform((), s_l, s_h) * area
        aspect_ratio = log_uniform((), r_1, r_2)
        h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

        if h < height and w < width:
            break

    x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
    y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

    padding_dims = [[y, height - h - y], [x, width - w - x]]
    mask = tf.pad(tf.zeros((h, w), dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, c])

    shelter = tf.random.uniform(shape, 0, max_random_value)
    shelter = tf.cast(shelter, image.dtype)
    image = tf.where(
        tf.equal(mask, 0), shelter, image)
    return image

@tf.function
def random_erasing2(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, max_random_value=1.0):

    if tf.random.uniform(()) > p:
        return image

    shape = tf.shape(image)
    height, width, c = shape[0], shape[1], shape[2]
    area = tf.cast(height * width, tf.float32)

    while tf.constant(True):
        target_area = tf.random.uniform((), s_l, s_h) * area
        aspect_ratio = log_uniform((), r_1, r_2)
        h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

        if h < height and w < width:
            break

    x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
    y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

    top = image[0:y, :, :]
    mid_left = image[y:y+h, 0:x, :]
    mid_right = image[y:y+h, x+w:width, :]
    bottom = image[y+h:height, :, :]

    shelter = tf.random.uniform((h, w, c), 0, max_random_value)
    shelter = tf.cast(shelter, image.dtype)
    mid = tf.concat([mid_left, shelter, mid_right], 1)  # along x axis
    image = tf.concat([top, mid, bottom], 0)  # along y axis
    return image


@tf.function
def random_erasing3(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, max_random_value=1.0):

    if tf.random.uniform(()) > p:
        return image

    shape = tf.shape(image)
    height, width, c = shape[0], shape[1], shape[2]
    area = tf.cast(height * width, tf.float32)

    while tf.constant(True):
        target_area = tf.random.uniform((), s_l, s_h) * area
        aspect_ratio = log_uniform((), r_1, r_2)
        h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

        if h < height and w < width:
            break

    x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
    y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

    hi = tf.range(height)[:, None, None]
    mh = (hi >= y) & (hi < (y + h))
    wi = tf.range(width)[None, :, None]
    mw = (wi >= x) & (wi < (x + w))
    mask = tf.cast(tf.logical_not(mh & mw), image.dtype)

    shelter = tf.random.uniform(shape, 0, max_random_value)
    shelter = tf.cast(shelter, image.dtype)
    image = tf.where(
        tf.equal(mask, 0), shelter, image)
    return image



# %%timeit
image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
im1 = random_erasing(image)

# %%timeit
image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
im2 = random_erasing2(image)

# %%timeit
image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
im3 = random_erasing3(image)