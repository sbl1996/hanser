import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, RandomNormal

from hanser.model.layers import conv2d, bn, relu
from hanser.model.backbone.resnet import stack1


def flat(p, c):
    b = tf.shape(p)[0]
    ly, lx, o = p.shape[1:]
    p = tf.reshape(p, [b, ly * lx * (o // c), c])
    return p


class DetCat(Layer):

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, ps, **kwargs):
        loc_ps, cls_ps = ps
        loc_p = tf.concat([flat(p, 4) for p in loc_ps], axis=1)
        cls_p = tf.concat([flat(p, self.num_classes) for p in cls_ps], axis=1)
        return [loc_p, cls_p]

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ssd(backbone, num_extras=2, num_anchors=6, num_classes=21):

    cs = list(backbone.outputs)
    for i in range(num_extras):
        cs.append(stack1(cs[-1], 256, 1, stride1=2, name='extra_res%d' % (i + 1)))

    loc_ps = [
        conv2d(c, num_anchors * 4, kernel_size=3, use_bias=True,
               kernel_initializer=RandomNormal(0, 0.01),
               name='loc_head%d' % (i + 3))
        for i, c in enumerate(cs)
    ]

    cls_ps = [
        conv2d(c, num_anchors * num_classes, kernel_size=3, use_bias=True,
               kernel_initializer=RandomNormal(0, 0.01),
               bias_initializer=Constant(-4.595),
               name='cls_head%d' % (i + 3))
        for i, c in enumerate(cs)
    ]

    loc_p, cls_p = DetCat(num_classes, name='detcat')([loc_ps, cls_ps])

    model = Model(inputs=backbone.inputs, outputs={'loc_p': loc_p, 'cls_p': cls_p})
    return model
