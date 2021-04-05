from tensorflow.keras import Sequential, Model

from hanser.models.cifar.resnet_vd import Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem


class ResNet(Model):

    def __init__(self, block, layers, zero_init_residual=False,
                 avd=False, stages=(64, 64, 128, 256, 512),
                 output_stride=32, multi_grad=(1, 1, 1)):
        super().__init__()
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0])
        self.in_channels = self.stages[0]

        strides = {
            8: (1, 2, 1, 1),
            16: (1, 2, 2, 1),
            32: (1, 2, 2, 2),
        }[output_stride]

        dilation = 1
        for i, (c, n, s) in enumerate(zip(stages[1:], layers, strides)):
            prev_dilation = dilation
            if i > 0 and s == 1:
                dilation *= 2
            dilations = [prev_dilation] + [dilation] * (n - 1)
            if i == 3:
                dilations = [m * d for m, d in zip(multi_grad, dilations)]

            layer = self._make_layer(
                block, channels=c, blocks=n, stride=s, dilations=dilations,
                zero_init_residual=zero_init_residual, avd=avd)
            setattr(self, "layer%d" % (i + 1), layer)

        self.feat_channels = [c * 4 for c in stages]

    def _make_layer(self, block, channels, blocks, stride=1, dilations=None, **kwargs):
        dilations = dilations or [1] * blocks
        layers = [block(self.in_channels, channels, stride=stride,
                        dilation=dilations[0], **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                dilation=dilations[i], **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)