from toolz import curry, get

import tensorflow as tf
from tensorflow.keras.metrics import Mean

import tensorflow_datasets as tfds

from hanser.tpu import setup
from hanser.datasets import prepare
from hanser.losses import smooth_l1_loss, cross_entropy_ohnm
from hanser.detection import match_anchors, detection_loss, batched_detect, coords_to_absolute
from hanser.detection.anchor import AnchorGenerator

from hanser.datasets.detection.voc import decode

from hanser.transform import normalize, photo_metric_distortion
from hanser.transform.detection import pad_to_fixed_size, random_hflip, resize, random_expand, random_sample_crop

from hanser.models.layers import set_defaults
from hanser.models.backbone.resnet_vd import resnet18
from hanser.models.detection.ssd import SSD
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
    scales_per_octave=1,
)
featmap_sizes = [
    [32, 32], [16, 16], [8, 8], [4, 4], [2, 2],
    # [64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2], [1, 1]
]

anchors = anchor_gen.grid_anchors(featmap_sizes)
flat_anchors = tf.concat(anchors, axis=0)

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=100, training=True):
    image, bboxes, labels, is_difficults, image_id = decode(example)
    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    # if training:
    #     image = photo_metric_distortion(image)
    #     image, bboxes = random_expand(image, bboxes, 4.0, mean_rgb)
    #     image, bboxes, labels, is_difficults = random_sample_crop(
    #         image, bboxes, labels, is_difficults)

    image = resize(image, output_size, keep_ratio=False)
    image = normalize(image, mean_rgb, std_rgb)

    # if training:
    #     image, bboxes = random_hflip(image, bboxes, 0.5)

    bboxes = coords_to_absolute(bboxes, tf.shape(image)[:2])
    box_t, cls_t, ignore = match_anchors(
        bboxes, labels, flat_anchors, pos_iou_thr=0.5, neg_iou_thr=0.5,
        bbox_std=(0.1, 0.1, 0.2, 0.2))

    bboxes = pad_to_fixed_size(bboxes, 0, [max_objects, 4])
    labels = pad_to_fixed_size(labels, 0, [max_objects])
    is_difficults = pad_to_fixed_size(is_difficults, 0, [max_objects])

    return image, {'box_t': box_t, 'cls_t': cls_t, 'ignore': ignore,
                   'bbox': bboxes, 'label': labels, 'is_difficult': is_difficults,
                   'image_id': image_id, }


mul = 1
n_train, n_val = 16, 16
batch_size, eval_batch_size = 4 * mul, 8
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

set_defaults({
    'bn': {
        'sync': True,
    }
})

backbone = resnet18()
model = SSD(backbone, anchor_gen.num_base_anchors, 21, extra_block_channels=(256, 128))
model.build((None, HEIGHT, WIDTH, 3))

# load_checkpoint()

criterion = detection_loss(box_loss=smooth_l1_loss, cls_loss=cross_entropy_ohnm(neg_pos_ratio=3.0))
base_lr = 1e-4
epochs = 60
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs, min_lr=0,
                       warmup_min_lr=base_lr, warmup_epoch=5)
optimizer = SGD(lr_schedule, momentum=0.9, nesterov=True, weight_decay=5e-4)

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
    return batched_detect(box_p, cls_p, flat_anchors, iou_threshold=0.45,
                          conf_threshold=0.02, conf_strategy='softmax',
                          bbox_std=(0.1, 0.1, 0.2, 0.2), label_offset=1)

learner.fit(
    ds_train, epochs, ds_val, val_freq=1,
    steps_per_epoch=steps_per_epoch, val_steps=val_steps,
    extra_metrics={'mAP': MeanAveragePrecision()},
    extra_output_transform=output_transform,
    extra_eval_freq=1,
)