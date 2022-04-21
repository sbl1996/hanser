import os
from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.detection import postprocess, coords_to_absolute, BBoxCoder, \
    max_iou_match, AnchorGenerator, DetectionLoss, l1_loss, focal_loss

from hanser.datasets.detection.coco import decode, make_dataset, label_map

from hanser.transform import normalize
from hanser.transform.detection import random_hflip, resize, pad_to, random_resize, random_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet import resnet50
from hanser.models.detection.retinanet import RetinaNet
from hanser.models.utils import load_pretrained_model

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics.common import MeanMetricWrapper
from hanser.train.metrics.detection import COCOEval, download_instances_val2017
from hanser.train.learner_v4 import SuperLearner

HEIGHT, WIDTH = (1024, 1024)

strides = [8, 16, 32, 64, 128]
anchor_gen = AnchorGenerator(
    strides=strides,
    ratios=[0.5, 1.0, 2.0],
    octave_base_scale=4,
    scales_per_octave=3,
)
featmap_sizes = [
    (HEIGHT // s, WIDTH // s) for s in strides
]

mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes)
anchors = tf.concat(mlvl_anchors, axis=0)

bbox_coder = BBoxCoder(anchors)


@curry
def transform(example, training=True):
    output_size = (HEIGHT, WIDTH)
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

    bbox_targets, labels, ignore = max_iou_match(
        gt_bboxes, gt_labels, bbox_coder, pos_iou_thr=0.5, neg_iou_thr=0.4)

    image = tf.cast(image, tf.bfloat16)
    return image, {'bbox_target': bbox_targets, 'label': labels, 'ignore': ignore, 'image_id': image_id}


mul = 2
batch_size, eval_batch_size = 16 * mul, 32 * mul
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
model = RetinaNet(backbone, anchor_gen.num_base_anchors[0], num_classes=80, norm='bn')
model.build((None, HEIGHT, WIDTH, 3))

load_pretrained_model("resnet50_nls", backbone)

criterion = DetectionLoss(
    box_loss_fn=l1_loss, cls_loss_fn=focal_loss(alpha=0.25, gamma=2.0))

base_lr = 0.01
epochs = 24
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_steps=500)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
}


def output_transform(output):
    bbox_preds, cls_scores = get(['bbox_pred', 'cls_score'], output)
    return postprocess(bbox_preds, cls_scores, bbox_coder,
                       iou_threshold=0.5, score_threshold=0.05, use_sigmoid=True)


ann_file = download_instances_val2017("./")
local_eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
    'AP': COCOEval(ann_file, (WIDTH, HEIGHT), output_transform,
                   label_transform=label_map),
}


learner = SuperLearner(
    model, criterion, optimizer, steps_per_loop=steps_per_epoch,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./drive/MyDrive/models/COCO-Detection/65")


learner.fit(
    ds_train_dist, epochs, ds_val_dist, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    local_eval_metrics=local_eval_metrics,
    local_eval_freq=[(0, 4), (16, 2)], save_freq=2,
)