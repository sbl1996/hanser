import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import ZeroPadding2D, Layer, Lambda, MaxPool2D, Input

from hanser.kerascv.efficientnet import efficientnet_b1b, efficientnet_b2b, efficientnet_b3b, efficientnet_b0b

from hanser.model.layers import conv2d, bn, relu


class WeightedSum(Layer):

    def __init__(self, shape, eps=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.eps = eps

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=self.shape,
            initializer='ones', dtype=tf.float32,
            trainable=True, name='alpha'
        )

    def call(self, inputs):
        w = tf.nn.relu(self.alpha)
        w = w / (tf.reduce_sum(w) + self.eps)
        p = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            p += inputs[i] * w[i]
        return p

    def get_config(self):
        config = super().get_config()
        config.update({'shape': self.shape, 'eps': self.eps})
        return config


def side_head(ps):
    size = ps[0].shape[1:3]
    ps = [
        bn(conv2d(p, 1, 1)) for p in ps
    ]
    ps = [
        ps[0],
        *[upsample(p, size) for p in ps[1:]]
    ]
    p = WeightedSum(len(ps))(ps)
    return p


def upsample(x, size):
    return Lambda(tf.compat.v1.image.resize_bilinear,
                  arguments=dict(
                      size=size,
                      align_corners=False,
                  ))(x)


def top_down_fusion2(p, pp, f_channels):
    pp = upsample(pp, p.shape[1:3])
    p = WeightedSum(2)([p, pp])
    p = conv2d(p, f_channels, kernel_size=3)
    p = bn(p)
    p = relu(p)
    return p


def bottom_up_fusion2(p, pp, f_channels):
    pp = MaxPool2D()(pp)
    p = WeightedSum(2)([p, pp])
    p = conv2d(p, f_channels, kernel_size=3)
    p = bn(p)
    p = relu(p)
    return p


def bottom_up_fusion3(p1, p2, pp, f_channels):
    pp = MaxPool2D()(pp)
    p = WeightedSum(3)([p1, p2, pp])
    p = conv2d(p, f_channels, kernel_size=3)
    p = bn(p)
    p = relu(p)
    return p


def bifpn(ps, f_channels):

    ps2 = [ps[-1]]
    for p in reversed(ps[:-1]):
        ps2.append(top_down_fusion2(p, ps2[-1], f_channels))
    ps3 = [ps2[-1]]
    ps2 = reversed(ps2[1:-1])

    for p1, p2 in zip(ps[1:-1], ps2):
        ps3.append(bottom_up_fusion3(p1, p2, ps3[-1], f_channels))
    ps3.append(bottom_up_fusion2(ps[-1], ps3[-1], f_channels))
    return tuple(ps3)


def eed(backbone='b2', in_size=(512, 512), f_channels=112, num_fpn_layers=4):
    if backbone == 'b0':
        backbone = efficientnet_b0b(in_size=in_size, pretrained=True, init_stride=1, strides_per_stage=[1, 2, 2, 2, 1])
    elif backbone == 'b1':
        backbone = efficientnet_b1b(in_size=in_size, pretrained=True, init_stride=1, strides_per_stage=[1, 2, 2, 2, 1])
    elif backbone == 'b2':
        backbone = efficientnet_b2b(in_size=in_size, pretrained=True, init_stride=1, strides_per_stage=[1, 2, 2, 2, 1])
    elif backbone == 'b3':
        backbone = efficientnet_b3b(in_size=in_size, pretrained=True, init_stride=1, strides_per_stage=[1, 2, 2, 2, 1])
    else:
        raise ValueError("Invalid backbone")
    c0 = backbone.get_layer('features/stage2/unit1/conv1/activ/mul').output
    c0 = ZeroPadding2D(((0, 1), (0, 1)), name='c0_padding')(c0)
    c1 = backbone.get_layer('features/stage3/unit1/conv1/activ/mul').output
    c2 = backbone.get_layer('features/stage4/unit1/conv1/activ/mul').output
    c3 = backbone.get_layer('features/final_block/activ/mul').output

    cs = [c0, c1, c2, c3]
    ps = [ bn(conv2d(c, f_channels, 1)) for c in cs ]
    pss = [
        [p] for p in ps
    ]
    for i in range(num_fpn_layers):
        ps = bifpn(ps, f_channels)
        for p, ps1 in zip(ps, pss):
            ps1.append(p)
    ps = [
        WeightedSum(num_fpn_layers + 1)(ps1)
        for ps1 in pss
    ]
    logits = side_head(ps)
    model = Model(inputs=backbone.inputs, outputs=logits)
    return model