from PIL import Image, ImageOps
import numpy as np

import tensorflow as tf
from hanser.datasets.segmentation.cityscapes import COLOR_MAP

from hanser.models.segmentation.backbone.resnet_vd import resnet50
from hanser.models.segmentation.deeplab import DeepLabV3P

from hanser.transform import resize
from hhutil.io import fmt_path

class SegModel:

    def __init__(self, size):
        height, width = size
        self.height = height
        self.width = width

    def preprocess(self, image):

        mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
        std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

        image = tf.cast(image, tf.float32)
        image = resize(image, (self.height, self.width), method='bilinear')

        image = (image - mean_rgb) / std_rgb
        return image

    def init(self, weights_path):
        backbone = resnet50(output_stride=16, multi_grad=(1, 2, 4))
        model = DeepLabV3P(backbone, aspp_ratios=(1, 6, 12, 18), aspp_channels=256, num_classes=19)
        model.build((None, self.height, self.width, 3))
        weights = np.load(weights_path, allow_pickle=True)
        model.set_weights(weights)
        self.model = model

    def get_pad(self, height, width):
        target_ratio = self.height / self.width
        h, w = height, width
        ratio = h / w
        if ratio > target_ratio:
            ph, pw = 0, h / target_ratio - w
        else:
            ph, pw = w * target_ratio - h, 0
        return int(ph), int(pw)

    def inference(self, img):
        w, h = img.size
        ph, pw = self.get_pad(h, w)
        img = ImageOps.pad(img, (w + pw, h + ph), centering=(0, 0))
        tw, th = img.size
        x = tf.convert_to_tensor(np.array(img))
        x = self.preprocess(x)
        y = self.model(x[None])[0]
        p = tf.argmax(y, axis=-1).numpy()
        seg = Image.fromarray(COLOR_MAP[p])
        seg = seg.resize((tw, th), resample=Image.NEAREST)
        seg = seg.crop((0, 0, w, h))
        return seg

model = SegModel(size=(512, 1024))
model.init(fmt_path("~/Downloads/Cityscapes-5.npy"))

img = Image.open(fmt_path("~/Downloads/bbac40d5711eca6d2a525f0179674468.jpeg"))
seg = model.inference(img)
seg.show()

import tensorflow_datasets as tfds