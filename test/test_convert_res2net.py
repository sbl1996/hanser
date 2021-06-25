from PIL import Image
import numpy as np
import tensorflow as tf

from hanser.distribute import setup_runtime

from hanser.models.imagenet.res2net.resnet_vd import resnet50
# from hanser.models.imagenet.res2net2.resnet_vd import resnet50
from hanser.models.utils import load_pretrained_model

setup_runtime(fp16=False)

img = Image.open("cat1.jpeg")
img = img.resize((457, 256), Image.BILINEAR).crop((116, 16, 116 + 224, 16 + 224))
x = np.array(img) / 255
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.astype(np.float32)

net = resnet50()
net.build((None, 224, 224, 3))
load_pretrained_model('res2netvd50', net, with_fc=True)

xt = tf.convert_to_tensor(x)
y1 = net(xt[None]).numpy()[0]