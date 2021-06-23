import tensorflow as tf

from hanser.models.imagenet.resnet_vd import resnet50
from hanser.models.utils import load_checkpoint

net = resnet50(num_classes=1000)
net.build((None, 224, 224, 3))

ckpt_path = '/Users/hrvvi/Downloads/ImageNet-83/model'
# reader = tf.train.load_checkpoint(ckpt_path)
load_checkpoint(ckpt_path, model=net)
net.save_weights("/Users/hrvvi/Downloads/ImageNet-83/model.h5")
net.load_weights("/Users/hrvvi/Downloads/ImageNet-83/model.h5", by_name=True, skip_mismatch=True)