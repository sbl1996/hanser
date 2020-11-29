import math
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


class FMix:
    r""" FMix augmentation
    
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
        Example
        ----------
        fmix = FMix(...)
        def loss(model, x, y, training=True):
            x = fmix(x)
            y_ = model(x, training=training)
            return tf.reduce_mean(fmix.loss(y_, y))
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):

        
        shape = [int(s) for s in x.shape][1:-1]
        lam, mask = sample_mask(self.alpha, self.decay_power, shape, self.max_soft, self.reformulate)
        index = np.random.permutation(int(x.shape[0]))
        index = tf.constant(index)
        mask = np.expand_dims(mask, -1)

        x1 = x * mask
        x2 = tf.gather(x, index) * (1 - mask)
        self.index = index
        self.lam = lam

        return x1 + x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)


def sample_mask(alpha, decay_power, shape):

    lam = tfp.distributions.Beta(alpha, alpha).sample(())

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape)

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

    param_size = [1] + list(freqs.shape) + [2]
    param = tf.random.normal(param_size)

    scale = scale[None, :, :, None]
    scale = tf.convert_to_tensor(scale, param.dtype)

    return scale * param


def make_low_freq_image(decay, shape):
    freqs = fftfreqnd(shape)
    spectrum = get_spectrum(freqs, decay, shape) #.reshape((1, *shape[:-1], -1))
    spectrum = tf.complex(spectrum[..., 0], spectrum[..., 1])
    mask = tf.signal.irfft2d(spectrum[0], shape)[None]

    mask = mask[:1, :shape[0], :shape[1]]

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