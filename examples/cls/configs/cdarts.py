from hanser.models.nas.genotypes import Genotype

seed = 42
mul = 1

# 2.55
CDARTS_cifar_2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3', 0), ('skip_connect', 1),
        ('sep_conv_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('skip_connect', 4)
    ],
    reduce_concat=[2, 3, 4, 5],
)

genotype = CDARTS_cifar_2

batch_size = 128
weight_decay = 5e-4
drop_path = 0.3

aux_weight = 0.4
base_lr = 0.025
epochs = 600
warmup_epoch = 10
grad_clip_norm = 5.0
sync_bn = mul == 1

val_freq = 5
valid_after = 500