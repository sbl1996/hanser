import numpy as np

import tensorflow as tf
from hanser.models.imagenet.res2net.resnet_vd import resnet50
from hanser.models.segmentation.backbone.res2net_vd import resnet50 as resnet50_backbone
from hanser.models.segmentation.deeplab import DeepLabV3

model = resnet50()
model.build((None, 224, 224, 3))
weights = model.get_weights()

backbone = resnet50_backbone(output_stride=16, multi_grad=(1, 2, 4))
model = DeepLabV3(backbone, aspp_ratios=(1, 6, 12, 18), aspp_channels=256, num_classes=19)
model.build((None, 384, 384, 3))

# ckpt = tf.train.Checkpoint(model=backbone)
# status = ckpt.read("/Users/hrvvi/Downloads/checkpoints/83/ckpt")
# status.assert_existing_objects_matched()

# x = tf.random.normal((2, 384, 384, 3))
# cs = model.backbone(x)
#
#
# weights = np.load("/Users/hrvvi/Downloads/resnet-50.npy", allow_pickle=True)
weights = weights[:-2]
model.backbone.set_weights(weights)
