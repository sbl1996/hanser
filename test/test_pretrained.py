from PIL import Image

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from hanser.datasets.imagenet_classes import IMAGENET_CLASSES

im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
im2 = im.resize((400, 224), resample=Image.BILINEAR)
im2 = im2.crop((88, 0, 88+224, 224))
x = np.array(im2)

net = ResNet50()
t = tf.cast(x, tf.float32)[None]
t = t[..., ::-1] - [103.939, 116.779, 123.68]

probs = net(t)
backbone = tf.keras.Model(net.input, net.get_layer("conv5_block3_out").output)
f = backbone(t)
ps, cs = tf.math.top_k(probs, k=5)
names = [IMAGENET_CLASSES[c] for c in cs.numpy()[0]]

from hanser.models.segmentation.backbone.resnet_keras import resnet_backbone
net2 = resnet_backbone(depth=50, output_stride=16)
fs = net2(t)

# import torch
# from torchvision.models import resnet50

# net2 = resnet50(pretrained=True)
# net2.eval()
# t2 = torch.from_numpy(x).to(torch.float32)[None].permute(0, 3, 1, 2)
# mean = torch.tensor([123.68, 116.779, 103.939])[None, :, None, None]
# std = torch.tensor([58.393, 57.12, 57.375])[None, :, None, None]
# t2 = (t2 - mean) / std
#
# logits = net2(t2)
# probs2 = torch.softmax(logits, dim=-1)
# ps2, cs2 = torch.topk(probs2, 5)