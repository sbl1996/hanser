import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Act


def interpolate(x, shape):
    dtype = x.dtype
    if dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if dtype != tf.float32:
        x = tf.cast(x, dtype)
    return x


class FPN(Layer):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)

    Example:
        >>> in_channels = [2, 3, 5]
        >>> scales = [64, 32, 16]
        >>> inputs = [tf.random.normal((1, s, s, c))
        >>>           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, 2)
        >>> outputs = self(inputs)
        >>> for i in range(len(outputs)):
        >>>     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_extra_convs=0):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_extra_convs = num_extra_convs

        self.lateral_convs = []
        self.fpn_convs = []
        self.extra_convs = []

        for in_channels in self.in_channels:
            l_conv = Conv2d(in_channels, out_channels, 1)
            fpn_conv = Conv2d(out_channels, out_channels, 3)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        in_channels = self.in_channels[-1]
        for i in range(num_extra_convs):
            extra_conv = Conv2d(in_channels, out_channels, 3, stride=2)
            if i != 0:
                extra_conv = Sequential([Act(), extra_conv])
            self.extra_convs.append(extra_conv)
            in_channels = out_channels

        self.feat_channels = [out_channels] * (len(self.in_channels) + num_extra_convs)

    def call(self, feats):
        laterals = [
            lateral_conv(feats[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        outs = []
        for i, fpn_conv in enumerate(reversed(self.fpn_convs)):
            x = laterals[-i-1]
            if i == 0:
                prev_x = x
            else:
                top_down_x = interpolate(prev_x, tf.shape(x)[1:3])
                prev_x = top_down_x + x
            x = fpn_conv(x)
            outs.insert(0, x)

        for extra_conv in self.extra_convs:
            outs.append(extra_conv(outs[-1]))
        return tuple(outs)
