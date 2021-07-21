# ResNet

## ResNet (D)
| Model         | Size | Augmentation | Epoch | Top 1 | Top 5 |   Time   |  Code  |  Log  |
|-------------- |:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|:-----:|
| ResNet_vd-50  |  160 |   Standard   |  120  | 78.54 | 94.33 |   192.1  | [50](/configs/ImageNet/50.py) | [1](/configs/ImageNet/log/50.log) [2](/configs/ImageNet/log/50-2.log) [3](/configs/ImageNet/log/50-3.log) |
| ResNet_vd-50  |  160 |     Mixup    |  200  | 78.93 | 94.68 |   214.1  | [31](/configs/ImageNet/31.py) | [1](/configs/ImageNet/log/31.log) |
| ResNet_vd-50  |  224 |   Standard   |  120  | 78.22 | 93.99 |   358.6  | [83](/configs/ImageNet/83.py) | [1](/configs/ImageNet/log/83.log) |
| ResNet_vd-50  |  224 |     Mixup    |  200  | 78.91 | 94.54 |   357.5  | [91](/configs/ImageNet/91.py) | [1](/configs/ImageNet/log/91.log) |

- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)


## RegNet
| Model         | Size | Augmentation | Epoch | Top 1 | Top 5 |  Time | Code |  Log  |
|---------------|:----:|:------------:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|
| RegNetY-1.6GF |  160 |   Standard   |  100  | 77.86 | 94.01 | 161.8 | [32](/configs/ImageNet/32.py) | [1](/configs/ImageNet/log/32.log) |
| RegNetY-4.0GF |  160 |   Standard   |  100  | 79.59 | 94.82 | 246.5 | [37](/configs/ImageNet/37.py) | [1](/configs/ImageNet/log/37.log) |
| RegNetY-1.6GF |  160 |     Mixup    |  200  | 78.38 | 94.27 | 164.0 | [36](/configs/ImageNet/36.py) | [1](/configs/ImageNet/log/36.log) |
| RegNetY-4.0GF |  160 |     Mixup    |  200  | 80.28 | 95.29 | 247.0 | [41](/configs/ImageNet/41.py) | [1](/configs/ImageNet/log/41.log) |

- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)


## Res2Net
| Model               | Size | Augmentation | Epoch | Top 1 | Top 5 |  Time | Code |  Log  |
|---------------------|:----:|:------------:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|
| Res2Net_vd-50       |  160 |   Standard   |  120  | 78.94 | 94.52 | 248.3 |  45  | [45](/configs/ImageNet/45.py) | [1](/configs/ImageNet/log/45.log) [2](/configs/ImageNet/log/45-2.log) |
| Res2Net_vd-50       |  160 |     Mixup    |  200  | 79.61 | 94.94 | 249.0 |  46  | [46](/configs/ImageNet/46.py) | [1](/configs/ImageNet/log/46.log) |
| Res2Net_vd-50       |  224 |   Standard   |  120  | 79.18 | 94.48 | 484.9 |  84  | [84](/configs/ImageNet/84.py) | [1](/configs/ImageNet/log/84.log) [2](/configs/ImageNet/log/84-2.log) |
| Res2Net_vd-50 (nlb) |  224 |   Standard   |  120  | 78.64 | 94.25 | 483.7 |  87  | [87](/configs/ImageNet/87.py) | [1](/configs/ImageNet/log/87.log) |

- [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)