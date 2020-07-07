import sys
import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from hanser.tpu import get_colab_tpu, auth

strategy = get_colab_tpu()

# auth()

from datetime import datetime
import math
import time
from toolz import curry

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy, MeanIoU, Accuracy

import hanser

from hanser.models.backbone.efficientnet import EfficientNetB0, DEFAULT_BLOCKS_ARGS
from hanser.legacy.models.segmentation.deeplab import deeplabv3

from hanser.train.trainer import Trainer
from hanser.datasets import prepare
from hanser.losses import cross_entropy
from hanser.train.lr_schedule import CosineDecayRestarts
from hanser.transform import resize, pad_to_bounding_box, _ImageDimensions
from hanser.transform.segmentation import random_crop, flip_dim, get_random_scale, random_scale
from hanser.datasets.tfrecord import parse_tfexample_to_img_seg

HEIGHT = WIDTH = 512


def decode(example_proto):
    example = parse_tfexample_to_img_seg(example_proto)
    img = tf.image.decode_image(example['image/encoded'])
    seg = tf.image.decode_image(example['image/segmentation/class/encoded'])
    img.set_shape([None, None, 3])
    seg.set_shape([None, None, 1])
    return img, seg


@curry
def preprocess(example, crop_h=HEIGHT, crop_w=WIDTH, min_size=None, ignore_label=255, training=True):
    img, seg = decode(example)

    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    img = tf.cast(img, tf.float32)
    if seg is not None:
        seg = tf.cast(seg, tf.int32)

    if min_size:
        img = resize(img, min_size)
        if seg is not None:
            seg = resize(seg, min_size, 'nearest')

    if training:
        scale = get_random_scale(0.5, 2.0, 0.25)
        img, seg = random_scale(img, seg, scale)

    img_h, img_w, c = _ImageDimensions(img, 3)
    target_h = img_h + tf.maximum(crop_h - img_h, 0)
    target_w = img_w + tf.maximum(crop_w - img_w, 0)

    img = pad_to_bounding_box(img, 0, 0, target_h, target_w, mean_rgb)
    if seg is not None:
        seg = pad_to_bounding_box(seg, 0, 0, target_h, target_w, 0)

    if training:
        img, seg = random_crop([img, seg], crop_h, crop_w)
        img, seg = flip_dim([img, seg], dim=1)

    img.set_shape([crop_h, crop_w, 3])
    seg.set_shape([crop_h, crop_w, 1])

    img = (img - mean_rgb) / std_rgb
    seg = tf.squeeze(seg, -1)
    return img, seg

# train_files = !gsutil ls -r gs://hrvvi-datasets/VOC2012Segmentation/trainaug* | cat
# val_files = !gsutil ls -r gs://hrvvi-datasets/VOC2012Segmentation/val* | cat
train_files = None
val_files = None

num_train_examples = 10582
num_val_examples = 1449
batch_size = 8 * 8
eval_batch_size = 20 * 8
steps_per_epoch = num_train_examples // batch_size
val_steps = num_val_examples // eval_batch_size
# test_steps = math.ceil(10000 / eval_batch_size)

ds_train = prepare(tf.data.TFRecordDataset(train_files), preprocess(training=True),
                   batch_size, training=True)
ds_val = prepare(tf.data.TFRecordDataset(val_files), preprocess(training=False),
                 eval_batch_size, training=False, drop_remainder=True)

ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
ds_val_dist = strategy.experimental_distribute_dataset(ds_val)

input_shape = (512, 512, 3)
model = deeplabv3(input_shape, 'resnet101', 16, multi_grad=(1, 2, 4), aspp=True, num_classes=21)
criterion = cross_entropy(ignore_label=255)
base_lr = 1e-3 * 8
lr_schedule = CosineDecayRestarts(base_lr, 100, steps_per_epoch=steps_per_epoch)
optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9, nesterov=True)

training_loss = Mean('loss', dtype=tf.float32)
training_accuracy = SparseCategoricalAccuracy('acc', dtype=tf.float32)
test_loss = Mean('loss', dtype=tf.float32)
test_accuracy = SparseCategoricalAccuracy('acc', dtype=tf.float32)


trainer = Trainer(model, criterion, optimizer,
                  metrics=[training_loss],
                  test_metrics=[test_loss],
                  model_dir="gs://hrvvi-models/checkpoints/deeplabv3-resnet50")


def output_transform(output):
    pred = tf.argmax(output, axis=-1, output_type=tf.int32)
#     pred = tf.image.resize(pred[..., None], (512, 512), method='nearest')[..., 0]
    return pred


def target_transform(target):
    return tf.where(tf.not_equal(target, 255), target, tf.zeros_like(target))

trainer.fit(
    100, ds_train_dist, steps_per_epoch, ds_val_dist, val_steps,
    extra_metrics=[MeanIoU(21, 'miou', dtype=tf.float32)],
    target_transform=target_transform,
    output_transform=output_transform,
    extra_eval_freq=5)