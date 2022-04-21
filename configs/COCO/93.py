import os
from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.detection import postprocess, coords_to_absolute, \
    atss_match, AnchorGenerator, GFLossV2, FCOSBBoxCoder

from hanser.datasets.detection.coco import decode, make_dataset, label_map

from hanser.transform import normalize
from hanser.transform.detection import random_hflip, random_resize, resize, pad_to, random_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet import resnet50
from hanser.models.detection.gfocal import GFocal
from hanser.models.utils import load_pretrained_model

from hanser.train.optimizers import AdamW
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics.common import MeanMetricWrapper
from hanser.train.metrics.detection import COCOEval, download_instances_val2017
from hanser.train.learner_v4 import SuperLearner

HEIGHT, WIDTH = (1024, 1024)

strides = [8, 16, 32, 64, 128]
anchor_gen = AnchorGenerator(
    strides=strides,
    ratios=[1.0],
    octave_base_scale=8,
    scales_per_octave=1,
)
featmap_sizes = [
    (HEIGHT // s, WIDTH // s) for s in strides
]

mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes)
num_level_bboxes = [a.shape[0] for a in mlvl_anchors]
anchors = tf.concat(mlvl_anchors, axis=0)

points = (anchors[:, :2] + anchors[:, 2:]) / 2
bbox_coder = FCOSBBoxCoder(points)

@curry
def transform(example, output_size=(HEIGHT, WIDTH), training=True):
    image, objects, image_id = decode(example)

    if training:
        image = random_resize(image, output_size, ratio_range=(0.5, 2.0))
        image, objects = random_crop(image, objects, output_size)
        image, objects = random_hflip(image, objects, 0.5)
    else:
        image = resize(image, output_size)

    image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    image, objects = pad_to(image, objects, output_size)

    gt_bboxes, gt_labels = get(['gt_bbox', 'gt_label'], objects)
    gt_bboxes = coords_to_absolute(gt_bboxes, tf.shape(image)[:2])

    bbox_targets, labels = atss_match(
        gt_bboxes, gt_labels, anchors, num_level_bboxes, topk=9, centerness=False)

    image = tf.cast(image, tf.bfloat16)
    return image, {'bbox_target': bbox_targets, 'label': labels, 'image_id': image_id}


mul = 4
batch_size, eval_batch_size = 16 * mul, 32 * 1
ds_train, ds_val, steps_per_epoch, val_steps = make_dataset(
    batch_size, eval_batch_size, transform, data_dir=os.getenv("REMOTE_DDIR"))

setup_runtime(fp16=True)
ds_train_dist, ds_val_dist = distribute_datasets(ds_train, ds_val)

set_defaults({
    'bn': {
        'sync': True,
    },
})

backbone = resnet50()
model = GFocal(backbone, num_classes=80, norm='bn')
model.build((None, HEIGHT, WIDTH, 3))

load_pretrained_model("resnet50_nls", backbone)

criterion = GFLossV2(
    bbox_coder, iou_loss_mode='giou', box_loss_weight=2.0)

base_lr = 0.0002
epochs = 50
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_epoch=1)
optimizer = AdamW(lr_schedule, weight_decay=0.05)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
}


def output_transform(output):
    bbox_preds, cls_scores = get(['bbox_pred', 'cls_score'], output)
    return postprocess(bbox_preds, cls_scores, bbox_coder, from_logits=False,
                       iou_threshold=0.6, score_threshold=0.05, use_sigmoid=True)


ann_file = download_instances_val2017("./")
local_eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
    'AP': COCOEval(ann_file, (WIDTH, HEIGHT), output_transform,
                   label_transform=label_map),
}


learner = SuperLearner(
    model, criterion, optimizer, steps_per_loop=steps_per_epoch,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./drive/MyDrive/models/COCO-Detection/93")


learner.fit(
    ds_train_dist, epochs, ds_val_dist, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    local_eval_metrics=local_eval_metrics,
    local_eval_freq=[(0, 8), (40, 2)],
)