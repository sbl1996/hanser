import resource
from toolz import curry

import tensorflow as tf
from hanser.datasets.cifar import make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor, cutout, mixup, mixup_batch, resizemix_batch
from hanser.transform.autoaugment import autoaugment

def get_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

used_mem = get_memory_usage()
print("used memory: {} Mb".format(used_mem / 1024 / 1024))

@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    # if training:
    #     image = cutout(image, 16)

    label = tf.one_hot(label, 100)

    return image, label


def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2)

def batch_transform(image, label):
    return resizemix_batch(image, label, scale=(0.1, 0.8), hard=True)

batch_size = 1024
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_cifar100_dataset(
    batch_size, eval_batch_size, transform, batch_transform=batch_transform)

used_mem = get_memory_usage()
print("used memory: {} Mb".format(used_mem / 1024 / 1024))

train_it = iter(ds_train)
for epoch in range(100):
    print(epoch)
    for i in range(steps_per_epoch):
        x, y = next(train_it)
        s = x.shape
        if i == steps_per_epoch - 1:
            used_mem = get_memory_usage()
            print("used memory: {} Mb".format(used_mem / 1024 / 1024))

# not reuse
# 801 -> 1006 -> 1011
# 802 -> 1006 -> 1011
# 802 -> 1006 -> 1011
# 802 -> 1007 -> 1010
# 800 -> 1004 -> 1009

# reuse
# 801
# 771
# 771
# 789
# 802

# mixup zip 548.8
# mixup batch 552.3