from hanser.models.nas.genotypes import Genotype

mul = 1

PC_DARTS_cifar = Genotype(
    normal=[
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 0),
        ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 3),
        ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)
    ], normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 0),
        ('max_pool_3x3', 2), ('sep_conv_5x5', 1),
        ('max_pool_3x3', 3), ('sep_conv_5x5', 2),
        ('sep_conv_5x5', 3), ('max_pool_3x3', 4)
    ], reduce_concat=range(2, 6)
)

genotype = PC_DARTS_cifar

batch_size = 96
weight_decay = 3e-4
drop_path = 0.3
aux_weight = 0.4
base_lr = 0.025
epochs = 620
warmup_epoch = 10
grad_clip_norm = 5.0
sync_bn = False

val_freq = 5
valid_after = 500