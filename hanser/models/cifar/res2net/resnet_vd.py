from hanser.models.common.res2net.resnet_vd import Bottle2neck
from hanser.models.cifar.resnet import _ResNet


class Res2Net(_ResNet):

    def __init__(self, depth, base_width=26, scale=4,
                 num_classes=10, channels=(64, 64, 128, 256)):
        block = Bottle2neck
        super().__init__(
            depth, block, num_classes, channels,
            base_width=base_width, scale=scale)