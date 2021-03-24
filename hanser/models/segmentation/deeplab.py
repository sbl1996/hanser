import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dropout
from hanser.models.layers import Conv2d, GlobalAvgPool

__all__ = ['DeepLabV3P', 'DeepLabV3']


def interpolate(x, shape):
    dtype = x.dtype
    if dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = tf.compat.v1.image.resize(
        x, shape, method=tf.compat.v1.image.ResizeMethod.BILINEAR, align_corners=False)
    if dtype != tf.float32:
        x = tf.cast(x, dtype)
    return x


class SeparableConv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 norm=None, act=None):
        super().__init__()
        self.depthwise_conv = Conv2d(
            in_channels, in_channels, kernel_size, groups=in_channels, dilation=dilation)
        self.piontwise_conv = Conv2d(
            in_channels, out_channels, kernel_size=1, norm=norm, act=act)

    def call(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ASPPModule(Layer):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 use_sep_conv=False,
                 image_pooling=False):
        super().__init__()

        self.blocks = []

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_op = SeparableConv2d
            else:
                conv_op = Conv2d
            block = conv_op(
                in_channels, out_channels, kernel_size=1 if ratio == 1 else 3, dilation=ratio,
                norm='def', act='def')
            self.blocks.append(block)

        out_size = len(self.blocks)

        if image_pooling:
            self.image_pooling = Sequential([
                GlobalAvgPool(keep_dim=True),
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def', act='def')
            ])
            out_size += 1
        else:
            self.image_pooling = None

        self.conv = Conv2d(out_channels * out_size, out_channels, kernel_size=1,
                           norm='def', act='def')

        # self.dropout = Dropout(p=0.1)  # drop rate

    def call(self, x):
        outputs = []
        interpolate_shape = tf.shape(x)[1:3]
        for block in self.blocks:
            y = block(x)
            y = interpolate(y, interpolate_shape)
            outputs.append(y)

        if self.image_pooling:
            y = self.image_pooling(x)
            y = interpolate(y, interpolate_shape)
            outputs.append(y)

        x = tf.concat(outputs, axis=-1)
        x = self.conv(x)
        # x = self.dropout(x)
        return x


class DeepLabV3P(Model):
    """
    The DeepLabV3Plus implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        backbone (Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_channels (int, optional): The output channels of ASPP module. Default: 256.
        num_classes (int): The unique number of target classes.
    """

    def __init__(self,
                 backbone,
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_channels=256,
                 num_classes=21):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in [0, 3]
        ]

        self.head = DeepLabV3PHead(backbone_channels, aspp_ratios,
                                   aspp_channels, num_classes)

    def call(self, x):
        img_shape = tf.shape(x)[1:3]
        feat_list = self.backbone(x)
        low_level_feat, x = feat_list[0], feat_list[3]
        logit = self.head(low_level_feat, x)
        return interpolate(logit, img_shape)


class DeepLabV3PHead(Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        backbone_channels (tuple|list): The same length with "backbone_indices". It indicates the channels of corresponding index.
        num_classes (int): The unique number of target classes.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_channels (int): The output channels of ASPP module.
    """

    def __init__(self, backbone_channels,
                 aspp_ratios, aspp_channels, num_classes):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_channels,
            use_sep_conv=True,
            image_pooling=True)
        self.decoder = Decoder(backbone_channels[0], aspp_channels, num_classes)

    def call(self, low_level_feat, x):
        x = self.aspp(x)
        logit = self.decoder(low_level_feat, x)
        return logit


class Decoder(Layer):

    def __init__(self, in_channels, aspp_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = Conv2d(in_channels, 48, kernel_size=1, norm='def', act='def')

        self.conv2 = SeparableConv2d(aspp_channels + 48, 256, kernel_size=3, norm='def', act='def')
        self.conv3 = SeparableConv2d(256, 256, kernel_size=3, norm='def', act='def')
        self.conv = Conv2d(256, num_classes, kernel_size=1)

    def call(self, low_level_feat, x):
        low_level_feat = self.conv1(low_level_feat)
        x = interpolate(x, tf.shape(low_level_feat)[1:3])
        x = tf.concat([x, low_level_feat], axis=-1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv(x)
        return x


class DeepLabV3(Model):
    """
    The DeepLabV3 implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf).

    Args:
        Please Refer to DeepLabV3P above.
    """

    def __init__(self,
                 backbone,
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_channels=256,
                 num_classes=21):
        super().__init__()

        self.backbone = backbone
        backbone_channels = backbone.feat_channels[-1]

        self.head = DeepLabV3Head(backbone_channels, aspp_ratios,
                                  aspp_channels, num_classes)

    def call(self, x):
        img_shape = tf.shape(x)[1:3]
        feat_list = self.backbone(x)
        logit = self.head(feat_list[-1])
        return interpolate(logit, img_shape)


class DeepLabV3Head(Layer):
    """
    The DeepLabV3Head implementation based on PaddlePaddle.

    Args:
        Please Refer to DeepLabV3PHead above.
    """

    def __init__(self, backbone_channels,
                 aspp_ratios, aspp_channels, num_classes):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels,
            aspp_channels,
            image_pooling=True)

        self.cls = Conv2d(aspp_channels, num_classes, kernel_size=1)

    def call(self, x):
        x = self.aspp(x)
        logit = self.cls(x)
        return logit