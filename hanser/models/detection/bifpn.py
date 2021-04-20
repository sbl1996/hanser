import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Act


class SeparableConv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, norm=None, act=None):
        super().__init__()
        self.depthwise_conv = Conv2d(
            in_channels, in_channels, kernel_size, groups=in_channels)
        self.piontwise_conv = Conv2d(
            in_channels, out_channels, kernel_size=1, norm=norm, act=act)

    def call(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


def interpolate(x, shape):
    dtype = x.dtype
    if dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = tf.compat.v1.image.resize_nearest_neighbor(x, shape)
    # x = tf.image.resize(x, shape, method=tf.image.ResizeMethod.BILINEAR)
    if dtype != tf.float32:
        x = tf.cast(x, dtype)
    return x


def maxpool(x):
    return tf.nn.max_pool2d(x, ksize=3, strides=2, padding='SAME')


class Resample(Layer):
    # TODO: Support both upsample and downsample

    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 1, norm=norm)\
            if in_channels != out_channels else Identity()

    def call(self, x):
        x = self.conv(x)
        x = maxpool(x)
        return x


def fast_fusion(xs, ws, eps=1e-4):
    dtype = xs[0].dtype
    ws = [tf.nn.relu(tf.cast(w, dtype)) for w in ws]
    ws_sum = tf.add_n(ws)
    xs = [
        xs[i] * ws[i] / (ws_sum + eps)
        for i in range(len(xs))
    ]
    x = tf.add_n(xs)
    return x


class BottomUpFusion2(Layer):
    def __init__(self, feat_channels, seperable_conv=True, norm='bn'):
        super().__init__()
        self.fusion_weights = [
            self.add_weight(
                f"weight{i+1}", shape=(), initializer='ones') for i in range(2)]
        conv_op = SeparableConv2d if seperable_conv else Conv2d
        self.conv = Sequential([
            Act(),
            conv_op(feat_channels, feat_channels, 3, norm=norm)
        ])

    def call(self, p, pp):
        pp = maxpool(pp)
        p = fast_fusion([p, pp], self.fusion_weights)
        p = self.conv(p)
        return p


class TopDownFusion2(Layer):
    def __init__(self, feat_channels, seperable_conv=True, norm='bn'):
        super().__init__()
        self.fusion_weights = [
            self.add_weight(
                f"weight{i+1}", shape=(), initializer='ones') for i in range(2)]

        conv_op = SeparableConv2d if seperable_conv else Conv2d
        self.conv = Sequential([
            Act(),
            conv_op(feat_channels, feat_channels, 3, norm=norm)
        ])

    def call(self, p, pp):
        h, w = p.shape[1:3]
        pp = interpolate(pp, (h, w))
        p = fast_fusion([p, pp], self.fusion_weights)
        p = self.conv(p)
        return p


class BottomUpFusion3(Layer):
    def __init__(self, feat_channels, seperable_conv=True, norm='bn'):
        super().__init__()
        self.fusion_weights = [
            self.add_weight(
                f"weight{i+1}", shape=(), initializer='ones') for i in range(3)]

        conv_op = SeparableConv2d if seperable_conv else Conv2d
        self.conv = Sequential([
            Act(),
            conv_op(feat_channels, feat_channels, 3, norm=norm)
        ])

    def call(self, p1, p2, pp):
        pp = maxpool(pp)
        p = fast_fusion([p1, p2, pp], self.fusion_weights)
        p = self.conv(p)
        return p


class BiFPNCell(Layer):
    def __init__(self, in_channels, feat_channels, seperable_conv=True, norm='bn'):
        super().__init__()
        n = len(in_channels)
        self.lats = [
            Conv2d(c, feat_channels, kernel_size=1, norm=norm)
            if c != feat_channels else Identity()
            for c in in_channels
        ]
        self.lats_c = [
            Conv2d(c, feat_channels, kernel_size=1, norm=norm)
            if c != feat_channels else Identity()
            for c in in_channels[1:-1]
        ]
        self.tds = [
            TopDownFusion2(feat_channels, seperable_conv, norm)
            for _ in range(n - 1)
        ]
        self.bus = [
            BottomUpFusion3(feat_channels, seperable_conv, norm)
            for _ in range(n - 2)
        ]
        self.bu = BottomUpFusion2(feat_channels, seperable_conv, norm)

    def call(self, ps):
        ps1 = [lat(p) for p, lat in zip(ps, self.lats)]

        ps2 = [ps1[-1]]
        for p, td in zip(reversed(ps1[:-1]), self.tds):
            ps2.append(td(p, ps2[-1]))
        ps3 = [ps2[-1]]
        ps2 = reversed(ps2[1:-1])

        for p, p2, lat, bu in zip(ps[1:-1], ps2, self.lats_c, self.bus):
            ps3.append(bu(lat(p), p2, ps3[-1]))
        ps3.append(self.bu(ps1[-1], ps3[-1]))
        return tuple(ps3)


class BiFPN(Layer):
    def __init__(self, in_channels, feat_channels, repeats, num_extra_levels=2, seperable_conv=True,
                 norm='bn'):
        super().__init__()
        in_channels = list(in_channels)
        self.extra_layers = []
        for i in range(num_extra_levels):
            self.extra_layers.append(Resample(in_channels[-1], feat_channels, norm))
            in_channels.append(feat_channels)

        self.cells = []
        for i in range(repeats):
            self.cells.append(
                BiFPNCell(in_channels, feat_channels, seperable_conv, norm))
            in_channels = [feat_channels] * len(in_channels)

    def call(self, ps):
        ps = list(ps)
        for l in self.extra_layers:
            ps.append(l(ps[-1]))
        for cell in self.cells:
            ps = cell(ps)
        return ps