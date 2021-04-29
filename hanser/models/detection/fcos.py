import tensorflow as tf
from tensorflow.keras import Model

from hanser.models.detection.detector import SingleStageDetector
from hanser.models.detection.neck.fpn import FPN
from hanser.models.detection.neck.bifpn import BiFPN
from hanser.models.detection.retinanet import RetinaHead


class FCOSBiFPN(Model):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=160, fpn_repeats=6, seperable_conv=False, fpn_act='def',
                 num_extra_levels=2, stacked_convs=4, norm='gn', centerness=True):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]

        self.neck = BiFPN(backbone_channels, feat_channels, fpn_repeats, num_extra_levels,
                          seperable_conv, norm, fpn_act)

        num_levels = len(backbone_indices) + num_extra_levels
        strides = [ 2 ** (l + backbone_indices[0] + 2) for l in range(num_levels) ]
        self.head = FCOSHead(
            num_classes, feat_channels, feat_channels, stacked_convs,
            strides=strides, norm=norm, centerness=centerness)


class FCOS(SingleStageDetector):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=256, num_extra_convs=2, stacked_convs=4,
                 norm='bn', centerness=True):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [backbone.feat_channels[i] for i in backbone_indices]
        self.neck = FPN(backbone_channels, feat_channels, num_extra_convs,
                        extra_convs_on='output', norm=norm)

        num_levels = len(backbone_indices) + num_extra_convs
        strides = [ 2 ** (l + backbone_indices[0] + 2) for l in range(num_levels) ]
        self.head = FCOSHead(
            num_classes, feat_channels, feat_channels, stacked_convs,
            strides=strides, norm=norm, centerness=centerness)


class FCOSHead(RetinaHead):

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4,
                 strides=(8, 16, 32, 64, 128), norm='bn', centerness=True):
        super().__init__(
            1, num_classes, in_channels, feat_channels, stacked_convs,
            centerness=centerness, concat=False, norm=norm, num_levels=len(strides))
        self.strides = strides

    def call(self, x):
        preds = super().call(x)
        bbox_preds = preds['bbox_pred']
        bbox_preds = [tf.nn.relu(bbox_preds[i]) * s for i, s in enumerate(self.strides)]
        preds = {**preds, "bbox_pred": bbox_preds}
        preds = {k: tf.concat(v, axis=1) for k, v in preds.items()}
        return preds