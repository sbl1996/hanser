from hanser.models.nas.genotypes import Genotype

seed = 0
mul = 1

PDARTS = Genotype(
    normal=[
        ('skip_connect', 0), ('dil_conv_3x3', 1),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0), ('dil_conv_3x3', 1),
        ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)
    ],
    reduce_concat=[2, 3, 4, 5]
)

genotype = PDARTS

batch_size = 96
weight_decay = 3e-4
drop_path = 0.3

aux_weight = 0.4
base_lr = 0.025
warmup_epoch = 0
epochs = 600 + warmup_epoch
grad_clip_norm = 5.0
sync_bn = False

val_freq = 1
valid_after = 500
