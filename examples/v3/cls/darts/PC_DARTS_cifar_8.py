# %%writefile config.py

from hanser.models.nas.genotypes import Genotype

seed = 0
mul = 8

# 2.57±0.07
PC_DARTS_cifar = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_5x5', 1), ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)
    ],
    reduce_concat=[2, 3, 4, 5]
)

genotype = PC_DARTS_cifar

batch_size = 96
weight_decay = 5e-4
drop_path = 0.3

aux_weight = 0.4
base_lr = 0.025
warmup_epoch = 10
epochs = 900 + warmup_epoch
grad_clip_norm = 5.0
sync_bn = False

val_freq = 1
valid_after = 500