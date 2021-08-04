from hanser.models.imagenet.resnet_vd import resnet50
from hanser.models.utils import load_pretrained_model

net = resnet50()
net.build((None, 224, 224, 3))
net.summary()

load_pretrained_model('resnetvd50', net, with_fc=False)