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
