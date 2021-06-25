# import math
# import tensorflow as tf
#
# from hanser.models.cifar.nasnet import NASNet
# from hanser.models.nas.genotypes import PC_DARTS_cifar, DARTS_V2
#
# net = NASNet(4, 5, True, 0.2, 10, DARTS_V2)
# net.build((None, 32, 32, 3))
#
# m1 = net.cells[0]._ops[0][0].layers[0].layers[1].layers[1].depthwise_kernel
# m2 = net.cells[0]._ops[0][0].layers[0].layers[5].layers[1].depthwise_kernel

from horch.models.layers import Conv2d as HorchConv2d
from hanser.models.layers import Conv2d

mt = HorchConv2d(128, 128, 3, groups=128, bias=True)
m = Conv2d(128, 128, 3, groups=128, bias=True)
m.build((None, 32, 32, 128))

print(mt.weight.std())
print(m.layers[1].depthwise_kernel.numpy().std())

print(mt.bias.std())
print(m.layers[1].bias.numpy().std())