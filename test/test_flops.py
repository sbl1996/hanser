from hanser.models.profile.fvcore import profile, count_mac
from hanser.models.imagenet.gen_resnet.res2net.resnet import resnet50
from hanser.models.imagenet.gen_resnet.resnet import resnet50
model = resnet50()
model.build((None, 224, 224, 3))
n, t = profile(model)

from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import Genotype

input_shape = (64, 64, 5)

my_genotype = Genotype(
    normal=[('avg_pool_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 1),
            ('skip_connect', 0), ('dil_conv_5x5', 2),
            ('skip_connect', 0), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1), ('avg_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3)],
    reduce_concat=range(2, 6)
)

model = NASNet(36, 23, True, 0.2, 4, my_genotype)
# model = NASNet(8, 8, True, 0.2, 4, my_genotype)
model.build((None, *input_shape))
profile(model)