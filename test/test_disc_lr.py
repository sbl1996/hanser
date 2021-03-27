import tensorflow_addons as tfa
from hanser.losses import cross_entropy
from hanser.models.segmentation.backbone.resnet_vd import resnet50
from hanser.models.segmentation.deeplab import DeepLabV3P
from hanser.train.lr_schedule import CosineLR
from hanser.train.optimizers import SGD

backbone = resnet50(output_stride=16, multi_grad=(1, 2, 4))
model = DeepLabV3P(backbone, aspp_ratios=(1, 6, 12, 18), aspp_channels=256, num_classes=21)

criterion = cross_entropy(ignore_label=255)
base_lr = 1e-2
epochs = 50

lr_schedule_fn = lambda lr: CosineLR(lr * 8, 20, epochs, min_lr=0,
                                     warmup_min_lr=lr, warmup_epoch=5)
optimizer_fn = lambda lr: SGD(lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
optimizer = tfa.optimizers.MultiOptimizer(([
    (optimizer_fn(lr_schedule_fn(base_lr * 0.1)), model.backbone),
    (optimizer_fn(lr_schedule_fn(base_lr)), model.head),
]))