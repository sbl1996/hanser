from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

import tensorflow_datasets as tfds

from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.detection import match_anchors, detection_loss, batched_detect, coords_to_absolute
from hanser.detection.anchor import AnchorGenerator

from hanser.transform import resize, photo_metric_distortion
from hanser.transform.detection import pad_to_fixed_size, random_hflip, random_sample_crop, random_expand, resize_and_pad

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet_vd import resnet50
from hanser.models.detection.retinanet import RetinaNet

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics import MeanMetricWrapper, MeanAveragePrecision
from hanser.train.cls import SuperLearner


HEIGHT = WIDTH = 512

anchor_gen = AnchorGenerator(
    strides=[8, 16, 32, 64, 128],
    ratios=[0.5, 1.0, 2.0],
    octave_base_scale=4,
    scales_per_octave=3,
)
featmap_sizes = [
    # [80, 80], [40, 40], [20, 20], [10, 10], [5, 5],
    [32, 32], [16, 16], [8, 8], [4, 4], [2, 2],
]

anchors = anchor_gen.grid_anchors(featmap_sizes)
flat_anchors = tf.concat(anchors, axis=0)

@curry
def preprocess(d, target_height=HEIGHT, target_width=WIDTH, max_objects=100, training=True):
    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    image_id = d['image/filename']
    str_len = tf.strings.length(image_id)
    image_id = tf.strings.to_number(
        tf.strings.substr(image_id, str_len - 10, 6),
        out_type=tf.int32
    )
    image_id = tf.where(str_len == 10, image_id + 10000, image_id)

    image = tf.cast(d['image'], tf.float32)
    bboxes, labels, is_difficults = d['objects']['bbox'], d['objects']['label'] + 1, d['objects']['is_difficult']
    labels = tf.cast(labels, tf.int32)

    if training:
    #     image = photo_metric_distortion(image)
    #     image, bboxes = random_expand(image, bboxes, 4.0, mean_rgb)
    #     image, bboxes, labels, is_difficults = random_sample_crop(
    #         image, bboxes, labels, is_difficults,
    #         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #         aspect_ratio_range=(0.5, 2.0))
    #     image = resize(image, (target_height, target_width))
        image, bboxes = random_hflip(image, bboxes, 0.5)
    # else:
    #     image = resize(image, (target_height, target_width))

    image, bboxes = resize_and_pad(image, bboxes, target_height, target_width, mean_rgb)
    image = (image - mean_rgb) / std_rgb

    bboxes = coords_to_absolute(bboxes, tf.shape(image)[:2])

    loc_t, cls_t, pos, ignore = match_anchors(
        bboxes, labels, flat_anchors, pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.)

    bboxes = pad_to_fixed_size(bboxes, 0, [max_objects, 4])
    labels = pad_to_fixed_size(labels, 0, [max_objects])
    is_difficults = pad_to_fixed_size(is_difficults, 0, [max_objects])

    return image, {'loc_t': loc_t, 'cls_t': cls_t, 'pos': pos, 'ignore': ignore,
                   'bbox': bboxes, 'label': labels, 'image_id': image_id, 'is_difficult': is_difficults}


mul = 1
n_train, n_val = 6, 4
batch_size, eval_batch_size = 2 * mul, 2
steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

ds_train = tfds.load("voc/2012", split=f"train[:{n_train}]",
               shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds_val = tfds.load("voc/2012", split=f"train[:{n_val}]",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds_train = prepare(ds_train, batch_size, preprocess(training=True),
                   training=True, repeat=False)
ds_val = prepare(ds_val, eval_batch_size, preprocess(training=False),
                 training=False, repeat=False, drop_remainder=True)

def output_transform(output):
    loc_p, cls_p = get(['loc_p', 'cls_p'], output)
    return batched_detect(loc_p, cls_p, flat_anchors, iou_threshold=0.5,
                          conf_threshold=0.05, conf_strategy='sigmoid')

m = MeanAveragePrecision()
m.reset_states()
for x, y in iter(ds_val):
    loc_p, cls_p = get(["loc_t", "cls_t"], y)
    cls_p = tf.one_hot(cls_p, 21, on_value=10.0, off_value=-10.0)[..., 1:]
    pred = output_transform({"loc_p": loc_p, "cls_p": cls_p})
    m.update_state(y, pred)
m.result()

from hanser.detection import random_colors
from PIL import Image
ds_val = tfds.load("voc/2012", split=f"train[2:3]",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
d = next(iter(ds_val))

target_height = HEIGHT
target_width = WIDTH
max_objects = 100
training = False

mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

image_id = d['image/filename']
str_len = tf.strings.length(image_id)
image_id = tf.strings.to_number(
    tf.strings.substr(image_id, str_len - 10, 6),
    out_type=tf.int32
)
image_id = tf.where(str_len == 10, image_id + 10000, image_id)

image = tf.cast(d['image'], tf.float32)
bboxes, labels, is_difficults = d['objects']['bbox'], d['objects']['label'] + 1, d['objects']['is_difficult']
labels = tf.cast(labels, tf.int32)

image, bboxes = resize_and_pad(image, bboxes, target_height, target_width, mean_rgb)
# im_b = tf.image.draw_bounding_boxes(image[None], bboxes[None], np.array(random_colors(bboxes.shape[0])) * 255)[0]
# Image.fromarray(im_b.numpy().astype(np.uint8)).show()
image = (image - mean_rgb) / std_rgb

bboxes = coords_to_absolute(bboxes, tf.shape(image)[:2])

# loc_t, cls_t, pos, ignore = match_anchors(
#     bboxes, labels, flat_anchors, pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.)
from hanser.detection import max_iou_assign, get_shape, bbox_encode, index_put, bbox_decode

gt_bboxes = bboxes
gt_labels = labels
anchors = flat_anchors
pos_iou_thr = 0.4
neg_iou_thr = 0.5
min_pos_iou = .0
bbox_std = (1., 1., 1., 1.)

assigned_gt_inds = max_iou_assign(anchors, gt_bboxes, pos_iou_thr, neg_iou_thr, min_pos_iou,
                                  match_low_quality=True, gt_max_assign_all=False)

num_anchors = get_shape(anchors, 0)

pos = assigned_gt_inds > 0
ignore = assigned_gt_inds == -1
indices = tf.range(num_anchors, dtype=tf.int32)[pos]

assigned_gt_inds = tf.gather(assigned_gt_inds, indices) - 1

assigned_gt_bboxes = tf.gather(gt_bboxes, assigned_gt_inds)
assigned_gt_labels = tf.gather(gt_labels, assigned_gt_inds)
assigned_anchors = tf.gather(anchors, indices)

loc_t = bbox_encode(assigned_gt_bboxes, assigned_anchors, bbox_std)
loc_t = index_put(
    tf.zeros([num_anchors, 4], dtype=tf.float32), indices, loc_t)
cls_t = index_put(
    tf.zeros([num_anchors, ], dtype=tf.int32), indices, assigned_gt_labels)

def output_transform(output):
    loc_p, cls_p = get(['loc_p', 'cls_p'], output)
    return batched_detect(loc_p, cls_p, flat_anchors, iou_threshold=0.6,
                          conf_threshold=0.05, conf_strategy='sigmoid')


m = MeanAveragePrecision()
m.reset_states()
loc_p, cls_p = loc_t[None], cls_t[None]
cls_p = tf.one_hot(cls_p, 21, on_value=10.0, off_value=-10.0)[..., 1:]
pred = output_transform({"loc_p": loc_p, "cls_p": cls_p})
m.update_state(y, pred)
m.result()
