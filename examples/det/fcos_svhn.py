from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.detection import postprocess, coords_to_absolute, FCOSBBoxCoder, \
    fcos_match, grid_points, DetectionLoss, focal_loss, iou_loss

from hanser.datasets.detection.svhn import decode, make_svhn_dataset_sub

from hanser.transform import normalize
from hanser.transform.detection import pad_to_fixed_size, random_hflip, random_resize, resize, pad_to, random_crop, \
    pad_objects

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet_vd import resnet10
from hanser.models.detection.fcos import FCOS
from hanser.models.utils import load_checkpoint

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics import MeanMetricWrapper, MeanAveragePrecision
from hanser.train.cls import SuperLearner


HEIGHT = WIDTH = 256

featmap_sizes = [
    [32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
strides = [8, 16, 32, 64, 128]
mlvl_points = grid_points(featmap_sizes, strides)
num_level_points = [p.shape[0] for p in mlvl_points]
points = tf.concat(mlvl_points, axis=0)

bbox_coder = FCOSBBoxCoder(points)

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=50, training=True):
    image, objects, image_id = decode(example)

    # if training:
    #     image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
    #     image, objects = random_crop(image, objects, output_size)
    #     image, objects = random_hflip(image, objects, 0.5)
    # else:
    image = resize(image, output_size)

    image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    image, objects = pad_to(image, objects, output_size)

    gt_bboxes, gt_labels = get(['gt_bbox', 'gt_label'], objects)
    gt_bboxes = coords_to_absolute(gt_bboxes, tf.shape(image)[:2])
    objects = {**objects, 'gt_bbox': gt_bboxes}

    bbox_targets, labels, centerness = fcos_match(
        gt_bboxes, gt_labels, points, num_level_points, strides=strides, radius=0.5)

    objects = pad_objects(objects, max_objects)

    return image, {'bbox_target': bbox_targets, 'label': labels, 'centerness': centerness,
                   **objects, 'image_id': image_id}

mul = 1
n_train, n_val = 16, 8
batch_size, eval_batch_size = 4 * mul, 4
ds_train, ds_val, steps_per_epoch, val_steps = make_svhn_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, preprocess)

backbone = resnet10()
model = FCOS(backbone, num_classes=20, feat_channels=64, stacked_convs=2, norm='bn')
model.build((None, HEIGHT, WIDTH, 3))

# load_checkpoint("./drive/MyDrive/models/ImageNet-86/ckpt", model=backbone)

criterion = DetectionLoss(
    box_loss_fn=iou_loss(mode='ciou', offset=True),
    cls_loss_fn=focal_loss(alpha=0.25, gamma=2.0), centerness=True)
base_lr = 0.0025
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

def output_transform(output):
    bbox_preds, cls_scores, centerness = get(['bbox_pred', 'cls_score', 'centerness'], output)
    return postprocess(bbox_preds, cls_scores, bbox_coder, centerness, nms_pre=1000,
                       iou_threshold=0.6, score_threshold=0.05, use_sigmoid=True)

local_eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
    'mAP': MeanAveragePrecision(output_transform=output_transform),
}


learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models")


learner.fit(
    ds_train, epochs, ds_val, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    local_eval_metrics=local_eval_metrics,
    local_eval_freq=[(0, 5), (45, 1)],
)