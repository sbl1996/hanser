from toolz import get
import random

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.detection import random_colors
from hanser.datasets.detection.voc import decode
from hanser.transform.detection import random_resize, random_crop, pad_to, random_hflip


ds_val = tfds.load("voc/2012", split=f"train[:100]",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds = [d for d in ds_val]

HEIGHT = WIDTH = 256
output_size = (HEIGHT, WIDTH)

ds1 = ds[:3]

d = random.choice(ds1)
image, bboxes, labels, is_difficults, image_id = decode(d)
image = random_resize(image, output_size, (0.8, 1.2))
image, bboxes = random_crop(image, bboxes, labels, is_difficults, output_size)
image, bboxes = random_hflip(image, bboxes)
# image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
image, bboxes = pad_to(image, bboxes, output_size)
im_b = tf.image.draw_bounding_boxes(image[None], bboxes[None], np.array(random_colors(bboxes.shape[0])) * 255)[0]
im = Image.fromarray(im_b.numpy().astype(np.uint8))
im.show()
