from toolz import curry, get

import tensorflow as tf

import tensorflow_datasets as tfds

from hanser.detection import match_anchors2, DetectionLoss, postprocess, coords_to_absolute, bbox_encode
from hanser.detection.assign import atss_assign
from hanser.detection.anchor import AnchorGenerator

from hanser.datasets.detection.voc import decode

from hanser.transform import normalize
from hanser.transform.detection import pad_to_fixed_size, random_hflip, random_resize, resize, pad_to, random_crop

HEIGHT = WIDTH = 640

anchor_gen = AnchorGenerator(
    strides=[8, 16, 32, 64, 128],
    ratios=[1.0],
    octave_base_scale=4,
    scales_per_octave=1,
)
featmap_sizes = [
    [80, 80], [40, 40], [20, 20], [10, 10], [5, 5],
]

anchors = anchor_gen.grid_anchors(featmap_sizes)
num_level_bboxes = [a.shape[0] for a in anchors]
flat_anchors = tf.concat(anchors, axis=0)

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=100, training=True):
    image, objects, image_id = decode(example)

    if training:
        image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
        image, objects = random_crop(image, objects, output_size)
        image, objects = random_hflip(image, objects, 0.5)
    else:
        image = resize(image, output_size)

    image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    image, objects = pad_to(image, objects, output_size)

    bboxes, labels, is_difficults = get(['bbox', 'label', 'is_difficult'], objects)
    bboxes = coords_to_absolute(bboxes, tf.shape(image)[:2])

    assigned_gt_inds = atss_assign(flat_anchors, num_level_bboxes, bboxes, topk=9)
    box_t, cls_t, ignore = match_anchors2(bboxes, labels, assigned_gt_inds)

    bboxes = pad_to_fixed_size(bboxes, max_objects)
    labels = pad_to_fixed_size(labels, max_objects)
    is_difficults = pad_to_fixed_size(is_difficults, max_objects)

    # image = tf.cast(image, tf.bfloat16)
    return image, {'box_t': box_t, 'cls_t': cls_t, 'ignore': ignore,
                   'bbox': bboxes, 'label': labels, 'is_difficult': is_difficults,
                   'image_id': image_id}

mul = 1
n_train, n_val = 16, 4
batch_size, eval_batch_size = 4 * mul, 4
steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

ds_train = tfds.load("voc/2012", split=f"train[:{n_train}]",
               shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds_val = tfds.load("voc/2012", split=f"train[:{n_val}]",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))

it = iter(ds_train)

output_size = (HEIGHT, WIDTH)
max_objects = 100
training = True

example = next(it)
image, objects, image_id = decode(example)

if training:
    image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
    image, objects = random_crop(image, objects, output_size)
    image, objects = random_hflip(image, objects, 0.5)
else:
    image = resize(image, output_size)

image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
image, objects = pad_to(image, objects, output_size)

bboxes, labels, is_difficults = get(['bbox', 'label', 'is_difficult'], objects)
bboxes = coords_to_absolute(bboxes, tf.shape(image)[:2])

assigned_gt_inds = atss_assign(flat_anchors, num_level_bboxes, bboxes, topk=9)
box_t, cls_t, ignore = match_anchors2(bboxes, labels, assigned_gt_inds)

pos = cls_t[None] != 0
bbox_encode(box_t[None], flat_anchors)[pos]