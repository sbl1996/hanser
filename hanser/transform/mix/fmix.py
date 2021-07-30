from toolz import curry
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


@curry
def fmix(data1, data2, alpha, decay_power):
    image1, label1 = data1
    image2, label2 = data2
    shape = image1.shape[:2]
    lam, mask = sample_mask(alpha, decay_power, shape)
    image = mask * image1 + (1 - mask) * image2

    lam = tf.cast(lam, label1.dtype)
    label = lam * label1 + (1 - lam) * label2
    return image, label


def sample_mask(alpha, decay_power, shape):

    lam = tfp.distributions.Beta(alpha, alpha).sample(())

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam)

    return lam, mask

def fftfreqnd(shape):
    h = shape[0]
    w = shape[1]
    fx = 0
    fy = np.fft.fftfreq(h)

    fy = np.expand_dims(fy, -1)

    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]

    return np.sqrt(fx * fx + fy * fy)


def get_spectrum(freqs, decay_power, shape):
    h = shape[0]
    w = shape[1]

    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h)])) ** decay_power)

    param_size = [2] + list(freqs.shape) + [1]
    param = tf.random.normal(param_size)

    scale = scale[None, :, :, None]
    scale = tf.convert_to_tensor(scale, param.dtype)

    return scale * param


def make_low_freq_image(decay, shape):
    freqs = fftfreqnd(shape)
    spectrum = get_spectrum(freqs, decay, shape) #.reshape((1, *shape[:-1], -1))
    spectrum = tf.complex(spectrum[0], spectrum[1])
    mask = tf.signal.irfft2d(spectrum[:, :, 0], shape)[:, :, None]

    mask = mask[:shape[0], :shape[1], :1]

    mask = (mask - tf.reduce_min(mask)) / tf.reduce_max(mask)
    return mask


def binarise_mask(mask, lam):
    mask_shape = mask.shape
    mask = tf.reshape(mask, [-1])
    idx = tf.argsort(mask, direction='DESCENDING')
    size = tf.cast(tf.shape(mask)[0], lam.dtype)
    num = tf.cond(
        tf.random.normal(()) > 0.5,
        lambda: tf.math.ceil(lam * size),
        lambda: tf.math.floor(lam * size))

    num = tf.cast(num, tf.int32)
    mask = tf.scatter_nd(idx[:num, None], tf.ones((num,)), mask.shape)
    mask = tf.reshape(mask, mask_shape)
    return mask