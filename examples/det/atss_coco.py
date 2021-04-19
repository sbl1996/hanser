from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.tpu import setup
from hanser.detection import postprocess, coords_to_absolute, BBoxCoder, \
    atss_match, AnchorGenerator, DetectionLoss, iou_loss, focal_loss

from hanser.datasets.detection.cocoval import decode, make_dataset_sub, label_map

from hanser.transform import normalize
from hanser.transform.detection import random_hflip, random_resize, resize, pad_to, random_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet_vd import resnet10
from hanser.models.detection.atss import ATSS
from hanser.models.utils import load_checkpoint

from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR
from hanser.train.metrics import MeanMetricWrapper, COCOEval
from hanser.train.cls import SuperLearner

HEIGHT = WIDTH = 256

strides = [8, 16, 32, 64, 128]
featmap_sizes = [
    (HEIGHT / s, WIDTH / s) for s in strides]

anchor_gen = AnchorGenerator(
    strides=strides,
    ratios=[1.0],
    octave_base_scale=2,
    scales_per_octave=1,
)

mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes)
num_level_bboxes = [a.shape[0] for a in mlvl_anchors]
anchors = tf.concat(mlvl_anchors, axis=0)

bbox_coder = BBoxCoder(anchors, (0.1, 0.1, 0.2, 0.2))

@curry
def transform(example, output_size=(HEIGHT, WIDTH), training=True):
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

    bbox_targets, labels, centerness = atss_match(
        gt_bboxes, gt_labels, anchors, num_level_bboxes, topk=9, centerness=True)

    return image, {'bbox_target': bbox_targets, 'label': labels, 'centerness': centerness,
                   'image_id': image_id}

mul = 1
n_train, n_val = 8, 8
batch_size, eval_batch_size = 4 * mul, 8

ds_train, ds_val, steps_per_epoch, val_steps = make_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, transform)

backbone = resnet10()
model = ATSS(backbone, num_classes=80, feat_channels=64, stacked_convs=2)
model.build((None, HEIGHT, WIDTH, 3))

criterion = DetectionLoss(
    box_loss_fn=iou_loss(mode='giou'), cls_loss_fn=focal_loss(alpha=0.25, gamma=2.0),
    bbox_coder=bbox_coder, decode_pred=True, centerness=True)

base_lr = 0.0025
epochs = 100
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=0, warmup_epoch=5)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=1e-4)

train_metrics = {
    'loss': Mean(),
}
eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
}

def output_transform(output):
    bbox_preds, cls_scores, centerness = get(
        ['bbox_pred', 'cls_score', 'centerness'], output, default=None)
    return postprocess(bbox_preds, cls_scores, bbox_coder, centerness, topk=100,
                       iou_threshold=0.6, score_threshold=0.05, use_sigmoid=True)

ann_file = "/Users/hrvvi/Downloads/annotations/instances_val2017.json"
local_eval_metrics = {
    'loss': MeanMetricWrapper(criterion),
    'AP': COCOEval(ann_file, (WIDTH, HEIGHT), output_transform,
                   label_transform=label_map),
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