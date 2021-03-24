import numpy as np

import tensorflow as tf
from hanser.models.segmentation.backbone.resnet_vd import resnet50
from hanser.models.segmentation.deeplab import DeepLabV3

backbone = resnet50(output_stride=8, multi_grad=(1, 2, 4))
model = DeepLabV3(backbone, aspp_ratios=(1, 6, 12, 18), aspp_channels=256, num_classes=19)
model.build((None, 384, 384, 3))

x = tf.random.normal((2, 384, 384, 3))
cs = model.backbone(x)


weights = np.load("/Users/hrvvi/Downloads/resnet-50.npy", allow_pickle=True)
weights = weights[:-2]
model.backbone.set_weights(weights)
