import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.imagenet import make_imagenet_dataset
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor

from hanser.train.optimizers import RMSprop
from hanser.models.imagenet.mobilenet_v3 import mobilenet_v3_large_125
from hanser.train.learner_v4 import SuperLearner
from hanser.train.metrics import TopKCategoricalAccuracy
from hanser.train.lr_schedule import ExponentialDecay
from hanser.losses import CrossEntropy
from hanser.train.callbacks import EMA

TASK_NAME = os.environ.get("TASK_NAME", "hworkflow-default")
TASK_ID = os.environ.get("TASK_ID", 0)
WORKER_ID = os.getenv("WORKER_ID", 0)

TRAIN_RES = 224

def transform(image, label, training):
    if training:
        image = random_resized_crop(image, TRAIN_RES, scale=(0.08, 1.0), ratio=(0.75, 1.33), fix=True)
        image = tf.image.random_flip_left_right(image)
    else:
        image = resize(image, 256)
        image = center_crop(image, 224)

    image, label = to_tensor(image, label, label_offset=1)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    label = tf.one_hot(label, 1000)
    return image, label

batch_size = 1024
eval_batch_size = 128

remote_dir, local_dir = os.getenv('REMOTE_DDIR'), os.getenv('LOCAL_DDIR')
ds_train, ds_eval, steps_per_epoch, eval_steps = make_imagenet_dataset(
    batch_size, eval_batch_size, transform, data_dir=(remote_dir, local_dir))


setup_runtime(fp16=True)
ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

model = mobilenet_v3_large_125(dropout=0.2)
model.build((None, TRAIN_RES, TRAIN_RES, 3))
model.summary()

criterion = CrossEntropy(label_smoothing=0.1)

base_lr = 0.016
epochs = 360
lr_schedule = ExponentialDecay(base_lr * (batch_size / 256), steps_per_epoch, 3.6, 0.97,
                               warmup_epoch=5, warmup_min_lr=0)
optimizer = RMSprop(lr_schedule, decay=0.9, momentum=0.9, epsilon=1e-3, weight_decay=1e-5)

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
    model, criterion, optimizer, steps_per_loop=steps_per_epoch,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./drive/MyDrive/models/{TASK_NAME}-{TASK_ID}-{WORKER_ID}")

if learner.load(miss_ok=True):
    learner.recover_log()

ema = EMA(0.9999, num_updates=optimizer.iterations)

learner.fit(ds_train, epochs, ds_eval, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=eval_steps,
            save_freq=10, callbacks=[ema])

ema.swap_weights()
learner.save()