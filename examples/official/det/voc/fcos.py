import os
from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.detection import postprocess, coords_to_absolute, DetectionLoss, \
    fcos_match, iou_loss, focal_loss, grid_points, FCOSBBoxCoder

from hanser.datasets.detection.voc import decode, make_voc_dataset

from hanser.transform import normalize
from hanser.transform.detection import pad_objects, random_hflip, random_resize, resize, pad_to, random_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet import resnet50
from hanser.models.detection.fcos import FCOS
from hanser.models.utils import load_pretrained_model

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics.common import MeanMetricWrapper
from hanser.train.metrics.detection import MeanAveragePrecision
from hanser.train.learner_v4 import SuperLearner

TASK_NAME = os.environ.get("TASK_NAME", "hstudio-default")
TASK_ID = os.environ.get("TASK_ID", 0)
WORKER_ID = os.getenv("WORKER_ID", 0)

HEIGHT = WIDTH = 640

featmap_sizes = [
    [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
strides = [8, 16, 32, 64, 128]
mlvl_points = grid_points(featmap_sizes, strides)
num_level_points = [p.shape[0] for p in mlvl_points]
points = tf.concat(mlvl_points, axis=0)

bbox_coder = FCOSBBoxCoder(points)

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=100, training=True):
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
    objects = {**objects, 'gt_bbox': gt_bboxes}

    bbox_targets, labels, centerness = fcos_match(
        gt_bboxes, gt_labels, points, num_level_points, strides=strides, radius=1.5)

    objects = pad_objects(objects, max_objects)

    return image, {'bbox_target': bbox_targets, 'label': labels, 'centerness': centerness,
                   **objects, 'image_id': image_id}

mul = 2
batch_size, eval_batch_size = 16 * mul, 64
ds_train, ds_val, steps_per_epoch, val_steps = make_voc_dataset(
    batch_size, eval_batch_size, preprocess, data_dir=os.getenv("REMOTE_DDIR"),
    drop_remainder=True)

setup_runtime(fp16=True)
ds_train_dist, ds_val_dist = distribute_datasets(ds_train, ds_val)

set_defaults({
    'bn': {
        'sync': True,
    },
})
backbone = resnet50()
model = FCOS(backbone, num_classes=20)
model.build((None, HEIGHT, WIDTH, 3))

load_pretrained_model("resnet50", backbone)

criterion = DetectionLoss(
    box_loss_fn=iou_loss(mode='ciou', offset=True),
    cls_loss_fn=focal_loss(alpha=0.25, gamma=2.0), centerness=True)
base_lr = 0.01
epochs = 50
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_epoch=3)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
}

def output_transform(output):
    bbox_preds, cls_scores, centerness = get(['bbox_pred', 'cls_score', 'centerness'], output)
    return postprocess(bbox_preds, cls_scores, bbox_coder, centerness,
                       iou_threshold=0.6, score_threshold=0.05, use_sigmoid=True)

local_eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
    'mAP': MeanAveragePrecision(output_transform=output_transform),
}


learner = SuperLearner(
    model, criterion, optimizer, steps_per_loop=steps_per_epoch,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./drive/MyDrive/models/{TASK_NAME}-{TASK_ID}-{WORKER_ID}")


learner.fit(
    ds_train_dist, epochs, ds_val_dist, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    local_eval_metrics=local_eval_metrics,
    local_eval_freq=[(0, 6), (18, 1)],
)