from itertools import repeat
from typing import Tuple, Iterable

import numpy as np
import tensorflow as tf


def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1, return_index=False):
    u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    gumbels = -tf.math.log(-tf.math.log(u))
    gumbels = (logits + gumbels) / tau
    y_soft = tf.nn.softmax(gumbels, axis=axis)
    if hard:
        index = tf.argmax(y_soft, axis=axis, output_type=tf.int32)
        y_hard = tf.one_hot(index, tf.shape(logits)[-1], dtype=logits.dtype)
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
        if return_index:
            ret = ret, index
    else:
        ret = y_soft
    return ret


def sample_relaxed_bernoulli(probs, temperature=1, hard=False):
    u = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)

    y_soft = tf.sigmoid(
        (tf.math.log(u) - tf.math.log1p(-u) +
         tf.math.log(probs) - tf.math.log1p(-probs)) / temperature)

    if hard:
        y_hard = tf.cast(tf.where(y_soft >= 0.5, 1.0, 0.0), probs.dtype)
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        ret = y_soft
    return ret


def nonzero(t):
    # assert t.ndim == 1 and t.dtype == tf.bool
    return tf.range(tf.shape(t)[0], dtype=tf.int32)[t]


def masked_scatter(t, mask, val):
    # assert t.dtype == val.dtype
    mask = tf.cast(mask, t.dtype)
    return tf.add(tf.multiply(t, 1 - mask), val * mask)


def index_put(t, indices, val):
    val = tf.cast(val, t.dtype)
    if val.shape.ndims == 0:
        val = tf.fill(tf.shape(indices), val)
    return tf.tensor_scatter_nd_update(t, indices[:, None], val)


def g(t, indices):
    return tf.gather(t, indices)


def to_float(x):
    return tf.cast(x, tf.float32)


def to_int(x):
    return tf.cast(x, tf.int32)


def choice(t, p=None):
    t = tf.convert_to_tensor(t)
    if p is None:
        p = tf.fill(t.shape, 1.0)
    p = to_float(p)[None]
    p = tf.math.log(p)
    i = tf.random.categorical(p, 1)[0, 0]
    return t[i]


def beta_mc(a, b, shape, mc_size=10000):
    mc_table = tf.constant(np.random.beta(a, b, mc_size), dtype=tf.float32)
    indices = tf.random.uniform(shape, 0, mc_size, dtype=tf.int32)
    return tf.gather(mc_table, indices)


def log_uniform(shape, minval, maxval, dtype=tf.float32):
    minval = tf.math.log(minval)
    maxval = tf.math.log(maxval)
    x = tf.random.uniform(shape, minval, maxval, dtype)
    return tf.exp(x)


def misc_concat(values):
    if isinstance(values, (tuple, list)):
        val = values[0]
        if tf.is_tensor(val):
            return tf.concat(values, 0)
        elif isinstance(val, dict):
            d = {}
            for k in val.keys():
                d[k] = misc_concat([v[k] for v in values])
            return d
        elif isinstance(val, (tuple, list)):
            return val.__class__(v for l in values for v in l)
        else:
            return values
    elif isinstance(values, dict):
        return {k: misc_concat(v) for k, v in values.items()}
    else:
        return values


def get_shape(tensor, axis):
    shape = tensor.shape[axis]
    if shape is None:
        return tf.shape(tensor)[axis]
    else:
        return shape


def triu(x, diag=True):
    y = tf.linalg.band_part(x, 0, -1)
    if not diag:
        y = y - tf.linalg.band_part(x, 0, 0)
    return y


def l2_norm(x, sqrt=False):
    if sqrt:
        return tf.norm(x, axis=-1)
    return tf.reduce_sum(tf.square(x), axis=-1)


def _pair(x) -> Tuple:
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))


def _meshgrid(x, y, row_major=False):
    xx, yy = tf.meshgrid(x, y, indexing='xy' if row_major else 'ij')
    xx, yy = tf.reshape(xx, (-1,)), tf.reshape(yy, (-1,))
    return xx, yy


def all_reduce(tensor, op):
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None:
        return tensor
    return replica_context.all_reduce(op, tensor)


def all_reduce_mean(tensor):
    return all_reduce(tensor, tf.distribute.ReduceOp.MEAN)


def all_reduce_sum(tensor):
    return all_reduce(tensor, tf.distribute.ReduceOp.SUM)


def safe_softmax(logits, axis):
    dtype = logits.dtype
    if dtype in [tf.float16, tf.bfloat16]:
        logits = tf.cast(logits, tf.float32)
        weights = tf.nn.softmax(logits, axis=axis)
        weights = tf.cast(weights, dtype)
    else:
        weights = tf.nn.softmax(logits, axis=axis)
    return weights


def prepend_dims(x, n):
    for i in range(n):
        x = tf.expand_dims(x, 0)
    return x


# def top_k(x, k):
#     NINF = tf.cast(-100000000, x.dtype)
#     results = []
#     v = tf.reduce_max(x, axis=-1, keepdims=True)
#     results.append(v)
#     for i in range(k - 1):
#         x = tf.where(x == v, NINF, x)
#         v = tf.reduce_max(x, axis=-1, keepdims=True)
#         results.append(v)
#     x = tf.concat(results, axis=-1)
#     return x


def top_k(x, k):
    """Equivalent to tf.math.top_k(x, k) but more efficient on tpu."""
    last_dim_size = x.shape[-1]
    min_value = tf.math.reduce_min(x) - 1.0

    out_values = []

    for unused_i in range(k):
        index = tf.math.argmax(x, axis=-1, output_type=tf.int32)
        mask = tf.one_hot(index, last_dim_size, dtype=x.dtype)
        # TODO(yonghui): Would tf.gather be more efficient and numerically stable here?
        value = tf.reduce_sum(mask * x, -1, keepdims=True)
        x = (1.0 - mask) * x + mask * min_value
        out_values.append(value)

    return tf.concat(out_values, -1)


def confusion_matrix(y_true, y_pred, num_classes):
    # Not work on TPU because of tf.math.bincount
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    c = num_classes
    return tf.reshape(tf.math.bincount(y_true * c + y_pred, minlength=c * c), (c, c))


def confusion_matrix_tpu(y_true, y_pred, num_classes, dtype=tf.int32):
    class_indices = tf.range(num_classes)
    tm = tf.equal(y_true[:, None], class_indices[None, :])
    pm = tf.equal(y_pred[:, None], class_indices[None, :])
    cm = tf.logical_and(tm[:, :, None], pm[:, None, :])
    cm = tf.reduce_sum(tf.cast(cm, dtype), axis=0)
    return cm


def in_top_k(predictions, targets, k):
    indices = tf.math.top_k(predictions, k=k, sorted=False).indices
    eq = tf.equal(targets[:, None], tf.cast(indices, targets.dtype))
    return tf.reduce_any(eq, axis=1)

# Inspired by tpu/models/official/mask_rcnn and torchvision.ops.roi_align
#
# https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/roi_align_kernel.cu
def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1.0, aligned=True):
    """Crop boxes from the image and resize them to crop_size.

    Args:
      input: A tensor with shape [batch, height, width, channels].
      boxes: A float tensor with shape [batch, num_boxes, 4]. The box coordinates
        in (y1, x1, y2, x2) format where the regions will be taken from. The
        coordinate must satisfy 0 <= x1 < x2 and 0 <= y1 < y2.
      output_size: The size of the output (in bins or pixels) after the pooling is performed.
      spatial_scale: A scaling factor that maps the box coordinates to the input coordinates.
        For example, if your boxes are defined on the scale of a 224x224 image and your input
        is a 112x112 feature map (resulting from a 0.5x scaling of the original image), you’ll
        want to set this to 0.5. Default: 1.0
      sampling_ratio: A number of sampling points in the interpolation grid used to compute the
        output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio
        sampling points per bin are used. If <= 0, then an adaptive number of grid points are used
        (computed as ceil(roi_width / output_width), and likewise for height). Default: 1
      aligned: If False, use the legacy implementation. If True, pixel shift the box coordinates
        it by -0.5 for a better alignment with the two neighboring pixel indices. This version is used in Detectron2

    Returns:
      A tensor with shape [batch, num_boxes, output_size, output_size, channels].
    """
    assert sampling_ratio == 1.0, 'Sampling is currently not supported'
    # Note that according to the Mask R-CNN paper,
    # "the results are not sensitive to the exact sampling locations,
    # or how many points are sampled, as long as no quantization is performed"
    # Therefore, sampling_ratio == 1.0 should work well.

    dtype = input.dtype
    shape = tf.shape(boxes)
    batch_size, num_boxes = shape[0], shape[1]

    shape = tf.shape(input)
    height, width, channels = shape[1], shape[2], shape[3]
    height = tf.cast(height, dtype)
    width = tf.cast(width, dtype)

    boxes = boxes * spatial_scale
    boxes_yx = boxes[..., 0:2]
    boxes_hw = boxes[..., 2:4] - boxes_yx

    grid = tf.range(output_size, dtype=dtype)
    box_grid_yx = boxes_yx[..., None, :] + (grid[:, None] + 0.5) * boxes_hw[..., None, :] / output_size

    if aligned:
        box_grid_yx = box_grid_yx - 0.5

    box_grid_yx0 = tf.floor(box_grid_yx)
    box_grid_yx0 = tf.maximum(0., box_grid_yx0)
    box_grid_yx1 = tf.minimum(box_grid_yx0 + 1, [height - 1, width - 1])

    # (batch_size, num_boxes, output_size * 2)
    box_gridx0x1 = tf.concat([box_grid_yx0[..., 1], box_grid_yx1[..., 1]], axis=-1)
    box_gridy0y1 = tf.concat([box_grid_yx0[..., 0], box_grid_yx1[..., 0]], axis=-1)

    indices_dtype = tf.int32
    x_indices = tf.cast(box_gridx0x1, indices_dtype)
    y_indices = tf.cast(box_gridy0y1, indices_dtype)

    height_dim_offset = width
    batch_dim_offset = height * height_dim_offset

    height_dim_offset = tf.cast(height_dim_offset, indices_dtype)
    batch_dim_offset = tf.cast(batch_dim_offset, indices_dtype)

    batch_dim_indices = tf.reshape(
        tf.range(batch_size, dtype=indices_dtype) * batch_dim_offset, [batch_size, 1, 1, 1])
    height_indices = tf.reshape(
            y_indices * height_dim_offset, [batch_size, num_boxes, output_size * 2, 1])
    width_indices = tf.reshape(
            x_indices, [batch_size, num_boxes, 1, output_size * 2])

    indices = batch_dim_indices + height_indices + width_indices

    indices = tf.reshape(indices, [-1])
    features = tf.reshape(input, [-1, channels])
    features_per_box = tf.gather(features, indices)

    features_per_box = tf.reshape(features_per_box,
        [batch_size, num_boxes, output_size * 2, output_size * 2, channels])

    # The RoIAlign feature f can be computed by bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    l_yx = box_grid_yx - box_grid_yx0
    h_yx = 1.0 - l_yx
    ly, lx, hy, hx = l_yx[..., 0], l_yx[..., 1], h_yx[..., 0], h_yx[..., 1]
    kernel_x = tf.reshape(
        tf.stack([hx, lx], axis=-1), [batch_size, num_boxes, 1, output_size * 2])
    kernel_y = tf.reshape(
        tf.stack([hy, ly], axis=-1), [batch_size, num_boxes, output_size * 2, 1])
    interpolation_kernel = kernel_y * kernel_x * 4

    features_per_box *= tf.cast(
        tf.expand_dims(interpolation_kernel, axis=-1),
        dtype=features_per_box.dtype)
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size * num_boxes, output_size * 2, output_size * 2, channels])
    features_per_box = tf.nn.avg_pool(features_per_box, [1, 2, 2, 1],
                                      [1, 2, 2, 1], 'VALID')
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size, num_boxes, output_size, output_size, channels])

    return features_per_box