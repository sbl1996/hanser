import os
from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.segmentation.cityscapes import make_dataset

from hanser.transform.segmentation import random_crop, flip_dim, random_scale
from hanser.transform import photo_metric_distortion

from hanser.models.utils import load_pretrained_model
from hanser.models.layers import set_defaults
from hanser.models.segmentation.backbone.resnet_vd import resnet50
from hanser.models.segmentation.deeplab import DeepLabV3P

from hanser.losses import cross_entropy
from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics.classification import CrossEntropy
from hanser.train.metrics.segmentation import MeanIoU
from hanser.train.learner_v4 import SuperLearner

HEIGHT, WIDTH = 512, 1024
IGNORE_LABEL = 255

@curry
def transform(image, label, training=True):
    crop_h, crop_w = HEIGHT, WIDTH

    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)

    if training:
        image, label = random_scale(image, label, (0.5, 2.0), 0.25)

        image, label = random_crop([image, label], (crop_h, crop_w))
        image, label = flip_dim([image, label], dim=1)
        image = photo_metric_distortion(image)

        image.set_shape([crop_h, crop_w, 3])
        label.set_shape([crop_h, crop_w, 1])
    else:
        image.set_shape([1024, 2048, 3])
        label.set_shape([1024, 2048, 1])

    image = (image - mean_rgb) / std_rgb
    label = tf.squeeze(label, -1)

    image = tf.cast(image, tf.bfloat16)
    return image, label

mul = 1
batch_size, eval_batch_size = 8 * mul, 8

ds_train, ds_val, steps_per_epoch, val_steps = make_dataset(
    batch_size, eval_batch_size, transform, data_dir=os.getenv('REMOTE_DDIR'),
    drop_remainder=True)

setup_runtime(fp16=True)
ds_train_dist, ds_val_dist = distribute_datasets(ds_train, ds_val)

set_defaults({
    'bn': {
        'sync': True,
    },
    'fixed_padding': False,
})

backbone = resnet50(output_stride=8, multi_grad=(1, 2, 4))
model = DeepLabV3P(backbone, aspp_ratios=(1, 12, 24, 36), aspp_channels=256, num_classes=19)
model.build((None, HEIGHT, WIDTH, 3))

load_pretrained_model("resnetvd50_nlb_fp", backbone)

criterion = cross_entropy(ignore_label=IGNORE_LABEL)
base_lr = 1e-2
epochs = 120
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_epoch=5)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': CrossEntropy(ignore_label=IGNORE_LABEL),
    'miou': MeanIoU(num_classes=19),
}

learner = SuperLearner(
    model, criterion, optimizer, steps_per_loop=steps_per_epoch,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models")

learner.fit(
    ds_train_dist, epochs, ds_val_dist, val_freq=5,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps)