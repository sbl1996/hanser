# Notice that weight names are different in checkpoint file and weights h5py file.
# Checkpoint: variable name (model/layer2/layer_with_weights-1/conv3/kernel)
# Weights: layer name (b'sequential_6, dense)
# Therefore, REMEMBER TO RUN THIS WITH A NEW PYTHON SESSION
# Otherwise, the layer name might be odd (sequential_1323, dense_16)
# and can't be used by transfer learning

import h5py

from hanser.models.imagenet.resnet_vd import resnet50
from hanser.models.utils import load_checkpoint

ckpt_path = '/Users/hrvvi/Downloads/ImageNet-83/ckpt'
weights_path = "/Users/hrvvi/Downloads/ImageNet-83/model.h5"

def call(self, x):
    x = self.stem(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    return x

net = resnet50()
net.build((None, 224, 224, 3))

load_checkpoint(ckpt_path, model=net)
net.save_weights(weights_path)

f = h5py.File(weights_path, 'r+')
f.attrs.keys()

def map_layer(f, layer_name, new_name):
    assert layer_name in f.keys()
