import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.imagenet import make_imagenet_dataset
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor, mixup_batch
from hanser.transform.autoaugment.imagenet import randaugment

from hanser.train.optimizers import SGDS
from hanser.models.imagenet.reresnet.beta import re_resnet_s
from hanser.train.metrics import TopKCategoricalAccuracy
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.models.defaults import set_defaults

TASK_NAME = os.environ.get("TASK_NAME", "hstudio-default")
TASK_ID = os.environ.get("TASK_ID", 0)
WORKER_ID = os.getenv("WORKER_ID", 0)

TRAIN_RES = 160

def transform(image, label, training):
    if training:
        image = random_resized_crop(image, TRAIN_RES, scale=(0.05, 1.0), ratio=(0.75, 1.33))
        image = tf.image.random_flip_left_right(image)
        image = randaugment(image, 2, 10)
    else:
        image = resize(image, 256)
        image = center_crop(image, 224)

    image, label = to_tensor(image, label, label_offset=1)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    label = tf.one_hot(label, 1000)
    return image, label


def batch_transform(image, label):
    return mixup_batch(image, label, 0.2)


batch_size = 256 * 3
eval_batch_size = 512

remote_dir, local_dir = os.getenv('REMOTE_DDIR'), os.getenv('LOCAL_DDIR')
ds_train, ds_eval, steps_per_epoch, eval_steps = make_imagenet_dataset(
    batch_size, eval_batch_size, transform, data_dir=(remote_dir, local_dir),
    batch_transform=batch_transform)

setup_runtime(fp16=True)
ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

set_defaults({
    'fixed_padding': True,
    'inplace_abn': {
        'enabled': True,
    }
})

model = re_resnet_s()
model.build((None, TRAIN_RES, TRAIN_RES, 3))
model.summary()

criterion = CrossEntropy(label_smoothing=0.1)

base_lr = 0.1
epochs = 300
lr_schedule = CosineLR(base_lr * (batch_size // 256), steps_per_epoch, epochs=epochs, min_lr=0,
                       warmup_epoch=5, warmup_min_lr=0)
optimizer = SGDS(lr_schedule, momentum=0.9, weight_decay=4e-5, nesterov=True)

train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
    'acc5': TopKCategoricalAccuracy(k=5),
}

learner = SuperLearner(
    model, criterion, optimizer, xla_compile=True,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./drive/MyDrive/models/{TASK_NAME}-{TASK_ID}-{WORKER_ID}")

learner.load("/content/drive/MyDrive/models/ImageNet-211-2/ckpt")


from hanser.datasets.imagenet import get_filenames, get_files, make_imagenet_dataset_split

res = []
# for test_res in [160, 192, 224, 256]:
for test_res in [224, 256, 288, 320, 352, 384]:
    for ratio in [0.875, 0.95]:

        def transform(image, label, training):
            image = resize(image, int(test_res / ratio))
            image = center_crop(image, test_res)

            image, label = to_tensor(image, label, label_offset=1)
            image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            label = tf.one_hot(label, 1000)
            return image, label

        eval_batch_size = 512

        eval_files = get_files(remote_dir, local_dir, get_filenames(training=False))
        ds_eval, eval_steps = make_imagenet_dataset_split(
            eval_batch_size, transform, eval_files, split='validation', training=False)

        ds_eval, = distribute_datasets(ds_eval)
        learner.evaluate(ds_eval, eval_steps)
        acc = learner.eval_metrics['acc'].result().numpy()
        res.append([test_res, ratio, acc])
        print("%s %.3f %.4f" % (test_res, ratio, acc))
for test_res, ratio, acc in res:
    print("%s %.3f %.4f" % (test_res, ratio, acc))