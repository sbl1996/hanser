from PIL import Image

import numpy as np

import tensorflow as tf

from hanser.models.imagenet.resnet_vd import resnet50
from hanser.models.utils import load_pretrained_model

net = resnet50()
net.build((None, 224, 224, 3))
load_pretrained_model('resnetvd50', net, with_fc=True)

save_path = "/Users/hrvvi/Downloads/resnet50"
tf.saved_model.save(net, save_path)

model = tf.saved_model.load(save_path)

img = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
img = img.resize((457, 256), Image.BILINEAR).crop((116, 16, 116 + 224, 16 + 224))
x = np.array(img) / 255
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = tf.convert_to_tensor(x, dtype=tf.float32)

y1 = net(x[None]).numpy()[0]
p1 = np.argpartition(y1, -5)[-5:]