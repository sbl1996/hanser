import numpy as np

import torch
import torch.nn.functional as F

import tensorflow as tf

def t2t(t):
    return torch.from_numpy(t.numpy())

def quality_focal_loss2(pred, target, beta=2.0):

    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    pos = (label > 0).nonzero().squeeze(1)
    pos_label = label[pos].long() - 1
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


def quality_focal_loss(y_true, y_pred, gamma=2.0):
    sigma = tf.sigmoid(y_pred)
    focal_weight = tf.abs(y_true - sigma) ** gamma
    losses = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    losses = losses * focal_weight
    return tf.reduce_sum(losses, axis=1)


C = 20
n = 100
pred = tf.random.normal((n, C))
label = tf.random.uniform((n,), minval=-8, maxval=C + 1, dtype=tf.int32)
label = tf.where(label < 0, 0, label)
pos = label > 0
score = tf.random.uniform((n,), minval=0, maxval=1)
score = tf.where(pos, score, 0)

pred1, label1, score1 = t2t(pred), t2t(label), t2t(score)
loss1 = quality_focal_loss2(pred1, (label1, score1))
target = tf.one_hot(label, C + 1)[..., 1:] * score[..., None]
loss = quality_focal_loss(target, pred)
np.testing.assert_allclose(loss.numpy(), loss1.numpy(), rtol=1e-6)