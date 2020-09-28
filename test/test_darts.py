from mxnet.gluon.data.vision.transforms import RandomLighting
from mxnet.symbol.gen_image import random_lighting

from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import PC_DARTS_cifar
net = NASNet(36, 20, True, 0, 10, PC_DARTS_cifar)
net.build((None, 32, 32, 3))
