from toolz import get
import random

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.detection import random_colors
from hanser.datasets.detection.voc import decode
from hanser.transform.detection import random_resize, random_crop, pad_to, random_hflip, resize

ds_val = tfds.load("voc/2012", split=f"train[:100]",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds = [d for d in ds_val]

HEIGHT = WIDTH = 512
output_size = (HEIGHT, WIDTH)

mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)

ds1 = ds[:10]

d = random.choice(ds1)

image, objects, image_id = decode(d)
image = resize(image, output_size)
# objects1 = objects
# image = photo_metric_distortion(image)
# image, objects = random_expand(image, objects, 4.0, mean_rgb)
# image, objects = random_sample_crop(image, objects)
# objects2 = objects
# image = resize(image, output_size, keep_ratio=False)
image, objects = random_hflip(image, objects)

# image = random_resize(image, output_size, (0.8, 1.2))
# image, objects = random_crop(image, objects, output_size)
# image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
image, objects = pad_to(image, objects, output_size, mode='random')
bboxes = objects['gt_bbox']
im_b = tf.image.draw_bounding_boxes(image[None], bboxes[None], np.array(random_colors(bboxes.shape[0])) * 255)[0]
im = Image.fromarray(im_b.numpy().astype(np.uint8))
im.show()
    # bboxes = objects['bbox']
    # for b in objects['bbox']:
    #     if tf.reduce_all(b == [0., 1., 0, 1]).numpy():
    #         br = True
    #         im_b = tf.image.draw_bounding_boxes(image[None], bboxes[None], np.array(random_colors(bboxes.shape[0])) * 255)[
    #             0]
    #         im = Image.fromarray(im_b.numpy().astype(np.uint8))
    #         im.show()
    #         break
    # i += 1