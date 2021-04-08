from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

import tensorflow_datasets as tfds

from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.losses import l1_loss, focal_loss
from hanser.detection import match_anchors, detection_loss, batched_detect, coords_to_absolute
from hanser.detection.anchor import AnchorGenerator

from hanser.datasets.detection.voc import decode

from hanser.transform import normalize
from hanser.transform.detection import pad_to_fixed_size, random_hflip, random_resize, resize, pad_to, random_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet_vd import resnet18
from hanser.models.detection.retinanet import RetinaNet
from hanser.models.utils import load_checkpoint

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics import MeanMetricWrapper, MeanAveragePrecision
from hanser.train.cls import SuperLearner


HEIGHT = WIDTH = 256

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
    box_t, cls_t, ignore = match_anchors(
        bboxes, labels, flat_anchors, pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.)

    bboxes = pad_to_fixed_size(bboxes, max_objects)
    labels = pad_to_fixed_size(labels, max_objects)
    is_difficults = pad_to_fixed_size(is_difficults, max_objects)

    # image = tf.cast(image, tf.bfloat16)
    return image, {'box_t': box_t, 'cls_t': cls_t, 'ignore': ignore,
                   'bbox': bboxes, 'label': labels, 'is_difficult': is_difficults,
                   'image_id': image_id}

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
# ds_train_dist, ds_val_dist = setup([ds_train, ds_val], fp16=True)

# set_defaults({
#     'bn': {
#         'sync': True,
#     }
# })
backbone = resnet18()
model = RetinaNet(backbone, anchor_gen.num_base_anchors[0], num_classes=20,
                  feat_channels=64, stacked_convs=2, use_norm=True)
model.build((None, HEIGHT, WIDTH, 3))

# load_checkpoint("./drive/MyDrive/models/ImageNet-86/ckpt", model=backbone)

criterion = detection_loss(box_loss=l1_loss, cls_loss=focal_loss(alpha=0.25, gamma=2.0, label_smoothing=0.1))
base_lr = 1e-3
epochs = 60
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
}


learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models")

def output_transform(output):
    box_p, cls_p = get(['box_p', 'cls_p'], output)
    return batched_detect(box_p, cls_p, flat_anchors, iou_threshold=0.5,
                          conf_threshold=0.05, conf_strategy='sigmoid')

learner.fit(
    ds_train, epochs, ds_val, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    extra_metrics={'mAP': MeanAveragePrecision()},
    extra_output_transform=output_transform,
    extra_eval_freq=1,
)