import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d
from hanser.models.cifar.resnet_vd import Bottleneck


class SSD(Model):

    def __init__(self, backbone, num_anchors, num_classes, extra_block_channels=(512,)):
        super().__init__()
        self.backbone = backbone

        backbone_channels = backbone.feat_channels[-3:]
        self.extra_blocks = []
        in_channels = backbone_channels[-1]
        for c in extra_block_channels:
            self.extra_blocks.append(Bottleneck(in_channels, c // 4, stride=2))

        self.head = SSDHead(backbone_channels + list(extra_block_channels),
                            num_anchors, num_classes)

    def call(self, x):
        xs = self.backbone(x)
        xs = list(xs[-3:])
        for block in self.extra_blocks:
            xs.append(block(xs[-1]))
        preds = self.head(xs)
        return preds


class SSDHead(Layer):

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and isinstance(num_anchors, (tuple, list))
        assert len(num_anchors) == len(in_channels)
        self.num_classes = num_classes
        self.in_channels = in_channels

        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                Conv2d(in_channels[i], num_anchors[i] * 4, kernel_size=3))
            cls_convs.append(
                Conv2d(in_channels[i], num_anchors[i] * num_classes, kernel_size=3))
        self.reg_convs = reg_convs
        self.cls_convs = cls_convs

    def call(self, feats):
        b = tf.shape(feats[0])[0]
        bbox_preds = []
        cls_scores = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            bbox_preds.append(tf.reshape(reg_conv(feat), [b, -1, 4]))
            cls_scores.append(tf.reshape(cls_conv(feat), [b, -1, self.num_classes]))
        box_p = tf.concat(bbox_preds, axis=1)
        cls_p = tf.concat(cls_scores, axis=1)
        return {'box_p': box_p, 'cls_p': cls_p}