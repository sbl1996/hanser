# Conclusion
# Type          ResNet  ResNet-β
# InplaceABN    1856    
# ReLU-BN

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.imagenet import make_imagenet_dataset
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor
from hanser.transform.autoaugment.imagenet import autoaugment

from hanser.train.optimizers import SGD
from hanser.models.imagenet.resnet_vd import resnet50
from hanser.train.cls import SuperLearner
from hanser.train.metrics import TopKCategoricalAccuracy
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.models.defaults import set_defaults

TRAIN_RES = 224

def transform(image, label, training):
    if training:
        image = random_resized_crop(image, TRAIN_RES, scale=(0.05, 1.0), ratio=(0.75, 1.33))
        image = tf.image.random_flip_left_right(image)
        image = autoaugment(image)
    else:
        image = resize(image, 256)
        image = center_crop(image, 224)

    image, label = to_tensor(image, label, label_offset=1)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    label = tf.one_hot(label, 1000)
    return image, label

batch_size = 1856
eval_batch_size = 128

train_files = ["./tfrecords/train-%05d-of-01024" % i for i in range(64)]
eval_files = ["./tfrecords/validation-%05d-of-00128" % i for i in range(8)]

ds_train, ds_eval, steps_per_epoch, eval_steps = make_imagenet_dataset(
    batch_size, eval_batch_size, transform, train_files=train_files, eval_files=eval_files)

setup_runtime(fp16=True)
ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

set_defaults({
    'fixed_padding': True,
    'inplace_abn': {
        'enabled': True,
    }
})

model = resnet50()
model.build((None, TRAIN_RES, TRAIN_RES, 3))
model.summary()

criterion = CrossEntropy(label_smoothing=0.1)

base_lr = 0.1
epochs = 20
lr_schedule = CosineLR(base_lr * (batch_size // 256), steps_per_epoch, epochs=epochs, min_lr=0,
                       warmup_epoch=5, warmup_min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=1e-4, nesterov=True)

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
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models/imagenet")
# learner.load(miss_ok=True)

learner.fit(ds_train, epochs, ds_eval, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=eval_steps,
            save_freq=5)