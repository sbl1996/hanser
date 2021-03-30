import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dropout
from hanser.models.layers import Conv2d, GlobalAvgPool, Identity

__all__ = ['DeepLabV3P', 'DeepLabV3']


def interpolate(x, shape):
    dtype = x.dtype
    if dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, shape, method=tf.image.ResizeMethod.BILINEAR)
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
                 image_pooling=False,
                 dropout_rate=0):
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

        self.dropout = Dropout(rate=dropout_rate) if dropout_rate else None

    def call(self, x):
        outputs = []
        interpolate_shape = tf.shape(x)[1:3]
        for block in self.blocks:
            y = block(x)
            outputs.append(y)

        if self.image_pooling:
            y = self.image_pooling(x)
            y = interpolate(y, interpolate_shape)
            outputs.append(y)

        x = tf.concat(outputs, axis=-1)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DeepLabV3P(Model):

    def __init__(self,
                 backbone,
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_channels=256,
                 aux_head=True,
                 num_classes=21):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in [0, 3]
        ]

        self.head = DeepLabV3PHead(backbone_channels, aspp_ratios,
                                   aspp_channels, num_classes)
        self.aux_head = FCNHead(backbone_channels[0], 1, 256,
                                dropout_rate=0.1, num_classes=num_classes) if aux_head else None

    def call(self, x):
        img_shape = tf.shape(x)[1:3]
        feat_list = self.backbone(x)
        low_level_feat, x = feat_list[0], feat_list[3]
        logits = self.head(low_level_feat, x)
        logits = interpolate(logits, img_shape)
        if self.aux_head:
            logits_aux = self.aux_head(x)
            logits_aux = interpolate(logits_aux, img_shape)
            return logits, logits_aux
        else:
            return logits


class DeepLabV3PHead(Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        in_channels (tuple|list): The same length with "backbone_indices". It indicates the channels of corresponding index.
        num_classes (int): The unique number of target classes.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_channels (int): The output channels of ASPP module.
    """

    def __init__(self, in_channels,
                 aspp_ratios, aspp_channels, num_classes):
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 2
        super().__init__()
        self.aspp = ASPPModule(
            aspp_ratios,
            in_channels[1],
            aspp_channels,
            use_sep_conv=True,
            image_pooling=True)
        self.decoder = Decoder(in_channels[0], aspp_channels, num_classes)

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

    def __init__(self,
                 in_channels,
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_channels=256,
                 num_classes=21):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            in_channels,
            aspp_channels,
            image_pooling=True)

        self.cls = Conv2d(aspp_channels, num_classes, kernel_size=1)

    def call(self, x):
        x = self.aspp(x)
        logit = self.cls(x)
        return logit


class FCNHead(Layer):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 num_convs=2,
                 channels=256,
                 kernel_size=3,
                 dilation=1,
                 dropout_rate=0.0,
                 num_classes=21):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        super(FCNHead, self).__init__()

        if num_convs == 0:
            self.convs = Identity()
        else:
            convs = []
            for i in range(num_convs):
                convs.append(
                    Conv2d(in_channels, channels, kernel_size=kernel_size, dilation=dilation,
                           norm='def', act='def'))
                in_channels = channels
            self.convs = Sequential(convs)
        self.dropout = Dropout(rate=dropout_rate) if dropout_rate else None

        self.cls = Conv2d(in_channels, num_classes, kernel_size=1)

    def call(self, x):
        x = self.convs(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.cls(x)
        return x
