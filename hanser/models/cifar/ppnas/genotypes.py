from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal')

PP_ResNet_CIFAR10_1 = Genotype(
    normal=[
        (1, 4, 'nor_conv_3x3'),
        (2, 3, 'nor_conv_3x3'),
        (1, 6, 'nor_conv_3x3'),
        (5, 7, 'nor_conv_3x3'),
    ],
)

PP_ResNet_CIFAR10_2 = Genotype(
    normal=[
        (3, 4, 'nor_conv_3x3'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
)

PP_ResNet_CIFAR10_3 = Genotype(
    normal=[
        (1, 3, 'nor_conv_3x3'),
        (3, 4, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (2, 7, 'nor_conv_3x3'),
    ],
)

PP_ResNet_CIFAR100_1 = Genotype(
    normal=[
        (3, 1, 'nor_conv_3x3'),
        (4, 5, 'nor_conv_3x3'),
        (6, 2, 'nor_conv_3x3'),
        (2, 7, 'nor_conv_3x3')
    ],
)

PP_ResNet_ImageNet_NS_1 = Genotype(
    normal=[
        [(2, 4, 'nor_conv_3x3'), (5, 1, 'nor_conv_3x3'), (6, 3, 'nor_conv_3x3'), (7, 6, 'nor_conv_3x3')],
        [(2, 1, 'nor_conv_3x3'), (5, 3, 'nor_conv_3x3'), (6, 1, 'nor_conv_3x3'), (7, 4, 'nor_conv_3x3')],
        [(2, 1, 'nor_conv_3x3'), (5, 3, 'nor_conv_3x3'), (6, 4, 'nor_conv_3x3'), (7, 4, 'nor_conv_3x3')],
        [(1, 4, 'nor_conv_3x3'), (3, 2, 'nor_conv_3x3'), (6, 4, 'nor_conv_3x3'), (2, 7, 'nor_conv_3x3')]
    ],
)

PP_ResNet_ImageNet_NS_2 = Genotype(
    normal=[
        [(2, 3, 'nor_conv_3x3'), (5, 1, 'nor_conv_3x3'), (6, 4, 'nor_conv_3x3'), (7, 6, 'nor_conv_3x3')],
        [(2, 4, 'nor_conv_3x3'), (5, 4, 'nor_conv_3x3'), (6, 1, 'nor_conv_3x3'), (7, 3, 'nor_conv_3x3')],
        [(2, 4, 'nor_conv_3x3'), (5, 3, 'nor_conv_3x3'), (6, 3, 'nor_conv_3x3'), (7, 1, 'nor_conv_3x3')],
        [(3, 1, 'nor_conv_3x3'), (2, 5, 'nor_conv_3x3'), (6, 2, 'nor_conv_3x3'), (4, 7, 'nor_conv_3x3')]
    ],
)

PP_ResNet_CIFAR10_FAIR_1 = Genotype(normal=[
    [
        (1, 2, 3, 4, 'nor_conv_3x3'),
        (3, 5, 'nor_conv_3x3'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (1, 3, 4, 5, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 4, 'nor_conv_3x3'),
        (1, 2, 3, 5, 'nor_conv_3x3'),
        (2, 5, 'nor_conv_3x3'),
        (1, 4, 6, 'nor_conv_3x3')
    ],
    [
        (1, 'nor_conv_3x3'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (4, 5, 6, 'nor_conv_3x3'),
        (1, 2, 4, 5, 7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR100_FAIR_1 = Genotype(normal=[
    [
        (2, 4, 'nor_conv_3x3'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (1, 2, 5, 6, 'nor_conv_3x3'),
        (2, 3, 4, 5, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 3, 'nor_conv_3x3'),
        (2, 4, 5, 'nor_conv_3x3'),
        (1, 3, 4, 5, 6, 'nor_conv_3x3'),
        (2, 4, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 3, 4, 'nor_conv_3x3'),
        (2, 4, 'nor_conv_3x3'),
        (1, 2, 3, 6, 'nor_conv_3x3'),
        (1, 2, 4, 6, 7, 'nor_conv_3x3')
    ]
])

PP_ResNet_ImageNet_FAIR_1 = Genotype(normal=[
    [
        (2, 4, 'skip_connect'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (1, 2, 5, 6, 'nor_conv_3x3'),
        (2, 3, 4, 5, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 3, 'skip_connect'),
        (2, 4, 5, 'nor_conv_3x3'),
        (1, 3, 4, 5, 6, 'nor_conv_3x3'),
        (2, 4, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 3, 4, 'skip_connect'),
        (2, 4, 'nor_conv_3x3'),
        (1, 2, 3, 6, 'nor_conv_3x3'),
        (1, 2, 4, 6, 7, 'nor_conv_3x3')
    ],
    [
        (2, 4, 'skip_connect'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (1, 2, 5, 6, 'nor_conv_3x3'),
        (2, 3, 4, 5, 6, 7, 'nor_conv_3x3')
    ],
])

PP_ResNet_ImageNet_T_1 = Genotype([
    [
        (3, 4, 'skip_connect'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
    [
        (3, 4, 'skip_connect'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
    [
        (3, 4, 'skip_connect'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
    [
        (3, 4, 'skip_connect'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
])

PP_ResNet_ImageNet_FAIR_AVD_1 = Genotype([
    [
        (2, 4, 'skip_connect'),
        (1, 3, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (3, 4, 'skip_connect'),
        (1, 2, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (2, 3, 'skip_connect'),
        (1, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 'skip_connect'),
        (3, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (3, 4, 7, 'nor_conv_3x3')
    ]
])

PP_ResNet_ImageNet_FAIR_AVD_2 = Genotype([
    [
        (2, 3, 'skip_connect'),
        (1, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 3, 4, 'skip_connect'),
        (1, 2, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 4, 'skip_connect'),
        (2, 3, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 'skip_connect'),
        (2, 3, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (4, 7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR10_FAIR_AVD_1 = Genotype([
    [
        (2, 3, 'skip_connect'),
        (1, 2, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (2, 3, 'skip_connect'),
        (1, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 3, 4, 'skip_connect'),
        (1, 2, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR10_FAIR_AVD_2 = Genotype([
    [
        (1, 4, 'skip_connect'),
        (2, 3, 4, 5, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 'skip_connect'),
        (2, 3, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (2, 4, 'skip_connect'),
        (1, 3, 5, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR100_FAIR_AVD_1 = Genotype([
    [
        (2, 3, 'skip_connect'),
        (1, 4, 5, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 4, 'skip_connect'),
        (2, 3, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 3, 'skip_connect'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR100_FAIR_AVD_2 = Genotype([
    [
        (4, 'skip_connect'),
        (1, 2, 3, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (6, 7, 'nor_conv_3x3')
    ],
    [
        (2, 3, 'skip_connect'),
        (1, 4, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ],
    [
        (1, 2, 4, 'skip_connect'),
        (3, 'nor_conv_3x3'),
        (6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ]
])


PP_ResNet_CIFAR_FAIR_AVD_R1 = Genotype([
    [
        (1, 2, 3, 'skip_connect'),
        (5, 'nor_conv_3x3'),
        (1, 2, 3, 4, 'nor_conv_3x3'),
        (3, 4, 'nor_conv_3x3')
    ],
    [
        (2, 3, 'skip_connect'),
        (5, 'nor_conv_3x3'),
        (2, 3, 6, 'nor_conv_3x3'),
        (1, 6, 7, 'nor_conv_3x3')
    ],
    [
        (3, 'skip_connect'),
        (1, 4, 5, 'nor_conv_3x3'),
        (2, 3, 5, 'nor_conv_3x3'),
        (1, 5, 7, 'nor_conv_3x3')
    ],
])

PP_ResNet_CIFAR_FAIR_AVD_R2 = Genotype([
    [
        (1, 2, 4, 'skip_connect'),
        (1, 2, 4, 5, 'nor_conv_3x3'),
        (1, 2, 5, 6, 'nor_conv_3x3'),
        (3, 6, 'nor_conv_3x3')
    ],
    [
        (1, 2, 3, 'skip_connect'),
        (2, 5, 'nor_conv_3x3'),
        (1, 4, 5, 'nor_conv_3x3'),
        (1, 2, 3, 4, 6, 7, 'nor_conv_3x3')
    ],
    [
        (1, 4, 'skip_connect'),
        (1, 2, 5, 'nor_conv_3x3'),
        (2, 4, 6, 'nor_conv_3x3'),
        (7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR_FAIR_AVD_RC1 = Genotype([
    [
        (1, 'skip_connect'),
        (1, 2, 3, 'nor_conv_3x3'),
        (1, 5, 'nor_conv_3x3'),
        (1, 2, 4, 5, 'nor_conv_3x3')
    ],
    [
        (4, 'skip_connect'),
        (1, 2, 3, 4, 'nor_conv_3x3'),
        (2, 3, 5, 'nor_conv_3x3'),
        (2, 3, 5, 6, 'nor_conv_3x3')
    ],
    [
        (3, 4, 'skip_connect'),
        (2, 4, 'nor_conv_3x3'),
        (1, 4, 6, 'nor_conv_3x3'),
        (3, 4, 5, 6, 7, 'nor_conv_3x3')
    ]
])

PP_ResNet_CIFAR_FAIR_AVD_RC2 = Genotype([
    [
        (1, 2, 'skip_connect'),
        (2, 4, 'nor_conv_3x3'),
        (1, 'nor_conv_3x3'),
        (3, 'nor_conv_3x3')
    ],
    [
        (2, 4, 'skip_connect'),
        (3, 4, 5, 'nor_conv_3x3'),
        (1, 3, 4, 6, 'nor_conv_3x3'),
        (1, 2, 5, 7, 'nor_conv_3x3')
    ],
    [
        (3, 'skip_connect'),
        (3, 4, 'nor_conv_3x3'),
        (1, 2, 4, 5, 6, 'nor_conv_3x3'),
        (4, 5, 'nor_conv_3x3')
    ]
])