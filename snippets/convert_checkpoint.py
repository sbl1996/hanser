from hanser.models.imagenet.resnet_vd import resnet50
from hanser.models.utils import load_checkpoint, convert_checkpoint

ckpt_path = '/Users/hrvvi/Downloads/ImageNet-83/ckpt'
fake_path = '/Users/hrvvi/Downloads/ImageNet-83/fake'

net = resnet50(num_classes=10)
net.build((None, 224, 224, 3))

load_checkpoint(fake_path, model=net)