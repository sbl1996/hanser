from tensorflow.keras import Sequential, Model

from hanser.models.cifar.resnet_vd import Bottleneck
from hanser.models.imagenet.stem import ResNetvdStem


class ResNet(Model):

    def __init__(self, block, layers, zero_init_residual=False,
                 avd=False, stages=(64, 64, 128, 256, 512),
                 output_stride=16, multi_grad=(1, 2, 4)):
        super().__init__()
        assert output_stride in [8, 16, 32]
        self.stages = stages

        self.stem = ResNetvdStem(self.stages[0])
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            zero_init_residual=zero_init_residual, avd=avd)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            zero_init_residual=zero_init_residual, avd=avd)

        prev_dilation = 1
        stride = 1 if output_stride == 8 else 2
        dilation = 2 if stride != 2 else 1
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=stride,
            zero_init_residual=zero_init_residual, avd=avd,
            dilations=[prev_dilation] + [dilation] * (layers[2] - 1),
        )

        prev_dilation = dilation
        stride = 1 if output_stride <= 16 else 2
        dilation *= 2 if stride != 2 else 1
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=stride,
            zero_init_residual=zero_init_residual, avd=avd,
            dilations=[prev_dilation * multi_grad[0]] + [m * dilation for m in multi_grad[1:]])

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