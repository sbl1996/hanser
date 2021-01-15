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