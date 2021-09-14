import math

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal, Constant, Zeros

from hanser.models.detection.detector import SingleStageDetector
from hanser.models.detection.neck.fpn import FPN
from hanser.models.detection.neck.bifpn import BiFPN
from hanser.models.layers import Conv2d, NormAct


class RetinaNetBiFPN(SingleStageDetector):

    def __init__(self, backbone, num_anchors, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=160, fpn_repeats=6, num_extra_levels=2, seperable_conv=False,
                 fpn_act='def', stacked_convs=4, norm='bn', centerness=False):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.neck = BiFPN(backbone_channels, feat_channels, fpn_repeats, num_extra_levels,
                          seperable_conv, norm, fpn_act)
        num_levels = len(backbone_indices) + num_extra_levels
        self.head = RetinaHead(num_anchors, num_classes, feat_channels, feat_channels, stacked_convs,
                               centerness=centerness, num_levels=num_levels, norm=norm)


class RetinaNet(SingleStageDetector):

    def __init__(self, backbone, num_anchors, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=256, extra_convs_on='input', num_extra_convs=2,
                 stacked_convs=4, norm='bn', centerness=False):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.neck = FPN(backbone_channels, feat_channels, num_extra_convs, extra_convs_on, norm)
        num_levels = len(backbone_indices) + num_extra_convs
        self.head = RetinaHead(num_anchors, num_classes, feat_channels, feat_channels, stacked_convs,
                               centerness=centerness, num_levels=num_levels, norm=norm)


class RetinaHead(Layer):

    def __init__(self, num_anchors, num_classes, in_channels, feat_channels, stacked_convs=4,
                 bbox_out_channels=None, centerness=False, concat=True, flatten=True,
                 num_levels=None, norm=None):
        super().__init__()
        if norm == 'bn':
            assert num_levels is not None
        if concat:
            assert flatten
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.bbox_out_channels = bbox_out_channels or 4
        self.centerness = centerness
        self.concat = concat
        self.flatten = flatten
        self.num_levels = num_levels
        self.norm = norm

        reg_convs = []
        cls_convs = []
        if self.norm == 'bn':
            reg_norm_acts = [[] for l in range(num_levels)]
            cls_norm_acts = [[] for l in range(num_levels)]
            for i in range(stacked_convs):
                reg_convs.append(
                    Conv2d(in_channels, feat_channels, 3,
                           kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))
                for l in range(num_levels):
                    reg_norm_acts[l].append(NormAct(feat_channels))

                cls_convs.append(
                    Conv2d(in_channels, feat_channels, 3,
                           kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))
                for l in range(num_levels):
                    cls_norm_acts[l].append(NormAct(feat_channels))

                in_channels = feat_channels
            self.reg_norm_acts = reg_norm_acts
            self.cls_norm_acts = cls_norm_acts
        else:
            for i in range(stacked_convs):
                reg_convs.append(
                    Conv2d(in_channels, feat_channels, 3, norm=norm, act='def',
                           kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))

                cls_convs.append(
                    Conv2d(in_channels, feat_channels, 3, norm=norm, act='def',
                           kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))

                in_channels = feat_channels
        self.reg_convs = reg_convs
        self.cls_convs = cls_convs

        if self.centerness:
            assert self.bbox_out_channels == 4
            self.bbox_out_channels += 1
        self.bbox_pred = Conv2d(
            feat_channels, num_anchors * self.bbox_out_channels, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros())

        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        self.cls_score = Conv2d(
            feat_channels, num_anchors * self.num_classes, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Constant(bias_value))

    # noinspection PyCallingNonCallable
    def call_single(self, x, i):
        reg_feat = x
        cls_feat = x
        for j, reg_conv in enumerate(self.reg_convs):
            reg_feat = reg_conv(reg_feat)
            if self.norm == 'bn':
                reg_feat = self.reg_norm_acts[i][j](reg_feat)
        for j, cls_conv in enumerate(self.cls_convs):
            cls_feat = cls_conv(cls_feat)
            if self.norm == 'bn':
                cls_feat = self.cls_norm_acts[i][j](cls_feat)
        bbox_pred = self.bbox_pred(reg_feat)
        cls_score = self.cls_score(cls_feat)
        return bbox_pred, cls_score

    def call(self, feats):
        b = tf.shape(feats[0])[0]
        bbox_preds = []
        cls_scores = []
        for i, x in enumerate(feats):
            h, w = x.shape[1:3]
            bbox_pred, cls_score = self.call_single(x, i)
            if self.flatten:
                a = self.num_anchors * h * w
                bbox_pred = tf.reshape(bbox_pred, [b, a, self.bbox_out_channels])
                cls_score = tf.reshape(cls_score, [b, a, self.num_classes])
            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)
        if self.concat:
            bbox_preds = tf.concat(bbox_preds, axis=1)
            cls_scores = tf.concat(cls_scores, axis=1)
        if self.centerness:
            if self.concat:
                centerness = bbox_preds[..., -1]
                bbox_preds = bbox_preds[..., :-1]
            else:
                centerness = [p[..., -1] for p in bbox_preds]
                bbox_preds = [p[..., :-1] for p in bbox_preds]
            return {'bbox_pred': bbox_preds, 'cls_score': cls_scores, 'centerness': centerness}
        else:
            return {'bbox_pred': bbox_preds, 'cls_score': cls_scores}