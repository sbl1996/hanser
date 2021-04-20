import tensorflow as tf
from tensorflow.keras import Model

from hanser.models.detection.fpn import FPN
from hanser.models.detection.bifpn import BiFPN
from hanser.models.detection.retinanet import RetinaHead, RetinaSepBNHead


class FCOSBiFPN(Model):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=160, fpn_repeats=6, seperable_conv=False, fpn_act='def',
                 stacked_convs=4, strides=(8, 16, 32, 64, 128), norm='gn'):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]

        num_extra_levels = len(strides) - len(backbone_indices)
        self.neck = BiFPN(backbone_channels, feat_channels, fpn_repeats, num_extra_levels, seperable_conv, norm, fpn_act)
        if norm == 'bn':
            self.head = FCOSSepBNHead(num_classes, feat_channels, feat_channels, stacked_convs, strides=strides)
        else:
            self.head = FCOSHead(
                num_classes, feat_channels, feat_channels, stacked_convs, strides=strides, norm=norm)

    def call(self, x):
        xs = self.backbone(x)
        xs = [xs[i] for i in self.backbone_indices]
        xs = self.neck(xs)
        preds = self.head(xs)
        return preds


class FCOS(Model):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=256, stacked_convs=4, strides=(8, 16, 32, 64, 128),
                 norm='gn'):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.neck = FPN(backbone_channels, feat_channels, 2,
                        extra_convs_on='output', norm=norm)
        if norm == 'bn':
            self.head = FCOSSepBNHead(num_classes, feat_channels, feat_channels, stacked_convs, strides=strides)
        else:
            self.head = FCOSHead(
                num_classes, feat_channels, feat_channels, stacked_convs, strides=strides, norm=norm)

    def call(self, x):
        xs = self.backbone(x)
        xs = [xs[i] for i in self.backbone_indices]
        xs = self.neck(xs)
        preds = self.head(xs)
        return preds


class FCOSHead(RetinaHead):

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4,
                 strides=(8, 16, 32, 64, 128), norm='gn'):
        super().__init__(
            1, num_classes, in_channels, feat_channels, stacked_convs,
            centerness=True, concat=False, norm=norm)
        self.strides = strides

    def call(self, x):
        preds = super().call(x)
        bbox_preds = preds['bbox_pred']
        bbox_preds = [ tf.nn.relu(bbox_preds[i]) * s for i, s in enumerate(self.strides)]
        preds = {**preds, "bbox_pred": bbox_preds}
        preds = { k: tf.concat(v, axis=1) for k, v in preds.items() }
        return preds


class FCOSSepBNHead(RetinaSepBNHead):

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4,
                 strides=(8, 16, 32, 64, 128)):
        super().__init__(
            1, num_classes, in_channels, feat_channels, stacked_convs, len(strides),
            centerness=True, concat=False)
        self.strides = strides

    def call(self, x):
        preds = super().call(x)
        bbox_preds = preds['bbox_pred']
        bbox_preds = [ tf.nn.relu(bbox_preds[i]) * s for i, s in enumerate(self.strides)]
        preds = {**preds, "bbox_pred": bbox_preds}
        preds = { k: tf.concat(v, axis=1) for k, v in preds.items() }
        return preds