from tensorflow.keras import Model, Sequential

from hanser.models.cifar.ppnas.resnet_avd_fair import Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem


class ResNet(Model):

    def __init__(self, genotype, block, layers, base_width=26, splits=4,
                 zero_init_residual=False, stages=(64, 64, 128, 256, 512),
                 output_stride=16, multi_grad=(1, 2, 4)):
        super().__init__()
        assert output_stride in [8, 16, 32]
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0])
        self.in_channels = self.stages[0]

        strides = {
            8: (1, 2, 1, 1),
            16: (1, 2, 2, 1),
            32: (1, 2, 2, 2),
        }[output_stride]

        genotype = genotype.normal

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
                base_width=base_width, splits=splits, genotype=genotype[i],
                zero_init_residual=zero_init_residual)
            setattr(self, "layer%d" % (i + 1), layer)

        self.feat_channels = [c * 4 for c in stages]

    def _make_layer(self, block, channels, blocks, stride, dilations=None, **kwargs):
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


def resnet50(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(genotype, **kwargs):
    return ResNet(genotype, Bottleneck, [3, 8, 36, 3], **kwargs)