import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal, Zeros

from hanser.ops import safe_softmax
from hanser.models.layers import Linear

from hanser.models.detection.detector import SingleStageDetector
from hanser.models.detection.neck.fpn import FPN
from hanser.models.detection.retinanet import RetinaHead

from hanser.detection.assign import mlvl_concat

def integral(prob):
    # n: (..., 4, n+1)
    reg_max = prob.shape[-1] - 1
    p = tf.constant(np.linspace(0, reg_max, reg_max + 1), dtype=prob.dtype)
    p = tf.tensordot(prob, p, axes=[[-1], [0]])
    return p


class GFocal(SingleStageDetector):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=256, num_extra_convs=2, stacked_convs=4, norm='bn'):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.neck = FPN(backbone_channels, feat_channels, num_extra_convs,
                        extra_convs_on='output', norm=norm)
        num_levels = len(backbone_indices) + num_extra_convs
        strides = [ 2 ** (l + backbone_indices[0] + 2) for l in range(num_levels) ]
        self.head = GFocalHead(
            num_classes, feat_channels, feat_channels, stacked_convs,
            norm=norm, strides=strides)


class GFocalHead(RetinaHead):

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4,
                 norm='bn', strides=(8, 16, 32, 64, 128), reg_max=16, reg_topk=4, reg_channels=64):
        super().__init__(
            1, num_classes, in_channels, feat_channels, stacked_convs,
            centerness=False, bbox_out_channels=4 * (reg_max + 1),
            concat=False, norm=norm, num_levels=len(strides))
        self.strides = strides
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels

        # self.reg_conf = Sequential([
        #     Linear(4 * (reg_topk + 1), reg_channels, act='relu',
        #            kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()),
        #     Linear(reg_channels, 1, act='sigmoid',
        #            kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()),
        # ])

    def call(self, x):
        preds = super().call(x)
        bbox_preds = preds['bbox_pred']
        cls_scores = preds['cls_score']

        b = tf.shape(bbox_preds[0])[0]
        num_level_bboxes = [p.shape[1] for p in bbox_preds]

        bbox_preds = tf.concat(bbox_preds, axis=1)
        cls_scores = tf.concat(cls_scores, axis=1)

        dis_logits = tf.reshape(bbox_preds, [b, -1, 4, self.reg_max + 1])
        prob = tf.nn.softmax(dis_logits, axis=-1)
        prob_topk = tf.math.top_k(prob, k=self.reg_topk).values
        stat = tf.concat([prob_topk, tf.reduce_mean(prob_topk, axis=-1, keepdims=True)], axis=-1)
        stat = tf.reshape(stat, [b, -1, 4 * (self.reg_topk + 1)])
        quality_score = tf.reduce_mean(stat, axis=-1, keepdims=True) * 3
        cls_scores = tf.nn.sigmoid(cls_scores) * quality_score

        scales = mlvl_concat(self.strides, num_level_bboxes, prob.dtype)[None, :, None]
        bbox_preds = integral(prob) * scales
        return {'dis_logit': dis_logits, 'bbox_pred': bbox_preds, 'cls_score': cls_scores,
                'scales': scales}