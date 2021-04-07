import math

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal, Constant, Zeros

from hanser.models.layers import Conv2d, Norm, Act
from hanser.models.detection.fpn import FPN


class RetinaNet(Model):

    def __init__(self, backbone, num_anchors, num_classes, feat_channels=256, stacked_convs=4, use_norm=False):
        super().__init__()
        self.backbone = backbone
        self.fpn = FPN(backbone.feat_channels[-3:], feat_channels, 2, use_norm)
        if use_norm:
            self.head = RetinaSepBNHead(feat_channels, feat_channels, stacked_convs, 5, num_anchors, num_classes)
        else:
            self.head = RetinaHead(feat_channels, feat_channels, stacked_convs, num_anchors, num_classes)

    def call(self, x):
        xs = self.backbone(x)
        xs = self.fpn(xs[-3:])
        preds = self.head(xs)
        return preds


class RetinaHead(Layer):

    def __init__(self, in_channels, feat_channels, stacked_convs, num_anchors, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        reg_convs = []
        cls_convs = []
        for i in range(stacked_convs):
            reg_convs.append(
                Conv2d(in_channels, feat_channels, 3, act='def',
                       kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))

            cls_convs.append(
                Conv2d(in_channels, feat_channels, 3, act='def',
                       kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))

            in_channels = feat_channels

        self.reg_convs = reg_convs
        self.cls_convs = cls_convs

        self.bbox_pred = Conv2d(
            feat_channels, num_anchors * 4, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros())

        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        self.cls_score = Conv2d(
            feat_channels, num_anchors * self.num_classes, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Constant(bias_value))


    def call_single(self, x):
        reg_feat = x
        cls_feat = x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        bbox_pred = self.bbox_pred(reg_feat)
        cls_score = self.cls_score(cls_feat)
        return bbox_pred, cls_score


    def call(self, feats):
        b = tf.shape(feats[0])[0]
        bbox_preds = []
        cls_scores = []
        for x in feats:
            bbox_pred, cls_score = self.call_single(x)
            bbox_preds.append(tf.reshape(bbox_pred, [b, -1, 4]))
            cls_scores.append(tf.reshape(cls_score, [b, -1, self.num_classes]))
        loc_p = tf.concat(bbox_preds, axis=1)
        cls_p = tf.concat(cls_scores, axis=1)
        return {'loc_p': loc_p, 'cls_p': cls_p}


class RetinaSepBNHead(Layer):

    def __init__(self, in_channels, feat_channels, stacked_convs, num_levels, num_anchors, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        reg_convs = []
        reg_norm_acts = [[] for l in range(num_levels)]
        cls_convs = []
        cls_norm_acts = [[] for l in range(num_levels)]
        for i in range(stacked_convs):
            reg_convs.append(
                Conv2d(in_channels, feat_channels, 3,
                       kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))
            for l in range(num_levels):
                reg_norm_acts[l].append(Sequential([Norm(feat_channels), Act()]))

            cls_convs.append(
                Conv2d(in_channels, feat_channels, 3,
                       kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros()))
            for l in range(num_levels):
                cls_norm_acts[l].append(Sequential([Norm(feat_channels), Act()]))

        self.reg_convs = reg_convs
        self.reg_norm_acts = reg_norm_acts
        self.cls_convs = cls_convs
        self.cls_norm_acts = cls_norm_acts

        self.bbox_pred = Conv2d(
            feat_channels, num_anchors * 4, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Zeros())

        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        self.cls_score = Conv2d(
            feat_channels, num_anchors * self.num_classes, 3,
            kernel_init=RandomNormal(stddev=0.01), bias_init=Constant(bias_value))

    def call_single(self, x, i):
        reg_feat = x
        cls_feat = x
        for reg_conv, norm_act in zip(self.reg_convs, self.reg_norm_acts[i]):
            reg_feat = reg_conv(reg_feat)
            reg_feat = norm_act(reg_feat)
        for cls_conv, norm_act in zip(self.cls_convs, self.cls_norm_acts[i]):
            cls_feat = cls_conv(cls_feat)
            cls_feat = norm_act(cls_feat)
        bbox_pred = self.bbox_pred(reg_feat)
        cls_score = self.cls_score(cls_feat)
        return bbox_pred, cls_score


    def call(self, feats):

        b = tf.shape(feats[0])[0]
        bbox_preds = []
        cls_scores = []
        for i, x in enumerate(feats):
            bbox_pred, cls_score = self.call_single(x, i)
            bbox_preds.append(tf.reshape(bbox_pred, [b, -1, 4]))
            cls_scores.append(tf.reshape(cls_score, [b, -1, self.num_classes]))
        loc_p = tf.concat(bbox_preds, axis=1)
        cls_p = tf.concat(cls_scores, axis=1)
        return {'loc_p': loc_p, 'cls_p': cls_p}
