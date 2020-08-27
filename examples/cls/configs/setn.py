from hanser.models.nas.genotypes import Genotype

seed = 42
mul = 8

SETN = Genotype(
    normal=[
        ('skip_connect', 0), ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 3),
        ('max_pool_3x3', 1), ('conv_3x1_1x3', 4),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0), ('skip_connect', 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

genotype = SETN

batch_size = 128
weight_decay = 5e-4
drop_path = 0.2
aux_weight = 0.4
base_lr = 0.025
epochs = 620
warmup_epoch = 10
grad_clip_norm = 5.0

val_freq = 2