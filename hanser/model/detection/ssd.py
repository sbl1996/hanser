import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU, Concatenate, AvgPool2D, UpSampling2D, Input, Lambda, Layer

from hanser.model.layers import conv2d, bn, relu
from hanser.model.backbone.resnet import stack1, load_weights, ResNet, block1


def flat(p, c):
    b = tf.shape(p)[0]
    ly, lx, o = p.shape[1:]
    p = tf.reshape(p, [b, ly * lx * (o // c), c])
    return p


class DetCat(Layer):

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def compute_output_shape(self, input_shapes):
        print(input_shapes)
        n = 0
        b = input_shapes[0][0]
        c = self.num_classes
        for s in input_shapes:
            ly, lx, o = s[1]
            n += ly * lx * (o // c)
        return [(b, n, 4), (b, n, c)]

    def call(self, ps):

        p = tf.concat([
            flat(p, 4 + self.num_classes)
            for p in ps
        ], axis=1)
        loc_p = p[:, :, :4]
        cls_p = p[:, :, 4:]
        return [loc_p, cls_p]

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ssd(backbone, num_extras=2, num_anchors=6, num_classes=21):
    # assert backbone.output_stride == 16
    # assert len(backbone.outputs) == 2

    cs = list(backbone.outputs)
    for i in range(num_extras):
        cs.append(stack1(cs[-1], 256, 1, stride1=2, name='conv%d' % (i + 6)))

    num_outputs = num_anchors * (4 + num_classes)
    bias_initializer = tf.keras.initializers.Constant(-4.595)
    ps = [
        conv2d(c, num_outputs, kernel_size=3, use_bias=True,
               bias_initializer=bias_initializer,
               name='pred%d' % (i + 3))
        for i, c in enumerate(cs)
    ]

    loc_p, cls_p = DetCat(num_classes, name='detcat')(ps)

    model = Model(inputs=backbone.inputs, outputs={'loc_p': loc_p, 'cls_p': cls_p})
    return model
