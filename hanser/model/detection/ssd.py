import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU, Concatenate, AvgPool2D, UpSampling2D, Input, Lambda, Layer

from hanser.model.layers import conv2d, bn, relu
from hanser.model.backbone.resnet import stack1, load_weights, ResNet, block1


def get_resnet(model_name, input_shape, pretrained=True, multi_grad=(1, 1, 1)):
    assert model_name in ['resnet50', 'resnet101']

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, {'resnet50': 6, 'resnet101': 23}[model_name], name='conv4')
        dilation = tuple(d * 2 for d in multi_grad)
        x = stack1(x, 512, 3,
                   stride1=1,
                   dilation=dilation,
                   name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, True, model_name)
    if pretrained:
        load_weights(model, model_name)

    return model


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


def ssd(input_shape, backbone, num_levels=4, multi_grad=(1, 1, 1), num_anchors=6, num_classes=21):
    assert backbone in ['resnet50', 'resnet101']
    backbone = get_resnet(backbone, input_shape, multi_grad=multi_grad)

    c3 = backbone.get_layer('conv4_block1_1_conv').input
    c4 = backbone.output
    cs = [c3, c4]
    for i in range(num_levels-2):
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
