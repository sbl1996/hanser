import numpy as np

import torch
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy as cross_entropy

def distribution_focal_loss2(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


def distribution_focal_loss(y_true, y_pred):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (tf.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (tf.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = tf.cast(y_true, tf.int32)
    dis_right = dis_left + 1
    weight_left = tf.cast(dis_right, tf.float32) - y_true
    weight_right = y_true - tf.cast(dis_left, tf.float32)
    loss = cross_entropy(dis_left, y_pred, from_logits=True) * weight_left \
        + cross_entropy(dis_right, y_pred, from_logits=True) * weight_right
    return loss

reg_max = 10
n = 1000
y_true = tf.random.uniform((n, 4), 0, reg_max - 1, dtype=tf.float32)
y_pred = tf.random.normal((n, 4, reg_max), dtype=tf.float32)
loss = distribution_focal_loss(y_true, y_pred)

y_true2 = torch.from_numpy(y_true.numpy())
y_pred2 = torch.from_numpy(y_pred.numpy())
loss2 = distribution_focal_loss2(y_pred2, y_true2)

np.testing.assert_allclose(loss.numpy(), loss2.numpy(), rtol=1e-6)