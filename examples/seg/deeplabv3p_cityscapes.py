from toolz import curry
import numpy as np

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.datasets.segmentation import decode
from hanser.datasets.segmentation.cityscapes import map_label

from hanser.transform import pad_to_bounding_box, _image_dimensions
from hanser.transform.segmentation import random_crop, flip_dim, get_random_scale, random_scale

from hanser.models.layers import set_defaults
from hanser.models.segmentation.backbone.resnet_vd import resnet50
from hanser.models.segmentation.deeplab import DeepLabV3P

from hanser.losses import cross_entropy
from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics import MeanIoU, CrossEntropy
from hanser.train.cls import SuperLearner

HEIGHT = WIDTH = 768
IGNORE_LABEL = 255

@curry
def preprocess(example, crop_h=HEIGHT, crop_w=WIDTH, ignore_label=IGNORE_LABEL, training=True):
    image, label = decode(example)
    label = tf.cast(label, tf.int32)
    label = map_label(label)

    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)

    if training:
        scale = get_random_scale(0.5, 2.0, 0.25)
        image, label = random_scale(image, label, scale)

        img_h, img_w, c = _image_dimensions(image, 3)
        target_h = tf.maximum(img_h, crop_h)
        target_w = tf.maximum(img_w, crop_w)

        image = pad_to_bounding_box(image, 0, 0, target_h, target_w, mean_rgb)
        label = pad_to_bounding_box(label, 0, 0, target_h, target_w, ignore_label)

        image, label = random_crop([image, label], crop_h, crop_w)
        image, label = flip_dim([image, label], dim=1)

        image.set_shape([crop_h, crop_w, 3])
        label.set_shape([crop_h, crop_w, 1])
    else:
        image.set_shape([1024, 2048, 3])
        label.set_shape([1024, 2048, 1])

    image = (image - mean_rgb) / std_rgb
    label = tf.squeeze(label, -1)

    image = tf.cast(image, tf.bfloat16)
    return image, label

train_files = [
    "/Users/hrvvi/Downloads/datasets/tfrecords/Cityscapes_sub/train-%05d-of-00004.tfrecord" % i for i in range(4)]
val_files = [
    "/Users/hrvvi/Downloads/datasets/tfrecords/Cityscapes_sub/val-%05d-of-00002.tfrecord" % i for i in range(2)]


mul = 8
n_train, n_val = 2975, 500
batch_size, eval_batch_size = 2 * mul, 2 * mul
steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

ds_train = prepare(tf.data.TFRecordDataset(train_files), batch_size, preprocess(training=True),
                   training=True, repeat=False)
ds_val = prepare(tf.data.TFRecordDataset(val_files), eval_batch_size, preprocess(training=False),
                 training=False, repeat=False)

# ds_train_dist, ds_val_dist = setup([ds_train, ds_val], fp16=True)

set_defaults({
    'bn': {
        'sync': True,
    }
})

backbone = resnet50(output_stride=16, multi_grad=(1, 2, 4))
model = DeepLabV3P(backbone, aspp_ratios=(1, 6, 12, 18), aspp_channels=256, num_classes=19)
model.build((None, HEIGHT, WIDTH, 3))

criterion = cross_entropy(ignore_label=255)
base_lr = 1e-3
epochs = 400
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_epoch=5)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=4e-5)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': CrossEntropy(ignore_label=255),
    'miou': MeanIoU(num_classes=19),
}

learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models")

learner.fit(
    ds_train_dist, epochs, ds_val_dist, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps)
