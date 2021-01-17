from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal')

PP_ResNet_1 = Genotype(
    normal=[
        (1, 4, 'nor_conv_3x3'),
        (2, 3, 'nor_conv_3x3'),
        (1, 6, 'nor_conv_3x3'),
        (5, 7, 'nor_conv_3x3'),
    ],
)

PP_ResNet_2 = Genotype(
    normal=[
        (3, 4, 'nor_conv_3x3'),
        (1, 5, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (1, 7, 'nor_conv_3x3'),
    ],
)

PP_ResNet_3 = Genotype(
    normal=[
        (1, 3, 'nor_conv_3x3'),
        (3, 4, 'nor_conv_3x3'),
        (2, 6, 'nor_conv_3x3'),
        (2, 7, 'nor_conv_3x3'),
    ],
)