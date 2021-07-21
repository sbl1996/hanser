# Assemble
Assemble means combining trcisk together to train better models.

## Setting 1
| Model            | Size |  Augmentation | Epoch | Top 1 | Top 5 |  Time |  Code  |  Log  |
|------------------|:----:|:-------------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| ResNet_vd-50     |  160 |     RA-AA     |  200  | 79.48 | 94.78 | 247.2 | [102](/configs/ImageNet/102.py) | [1](/configs/ImageNet/log/102.log) [2](/configs/ImageNet/log/102-2.log) [3](/configs/ImageNet/log/102-3.log) |
| SE_ResNet_vd-50  |  160 |     RA-AA     |  200  | 80.11 | 94.77 | 250.6 | [103](/configs/ImageNet/103.py) | [1](/configs/ImageNet/log/103.log) [2](/configs/ImageNet/log/103-2.log) [3](/configs/ImageNet/log/103-3.log) [3](/configs/ImageNet/log/103-4.log) [3](/configs/ImageNet/log/103-5.log) |
| SE_ResNet_vd-50  |  160 | TrivalAugment |  200  | 79.94 | 94.69 | 218.1 | [110](/configs/ImageNet/110.py) | [1](/configs/ImageNet/log/110.log) [2](/configs/ImageNet/log/110-2.log) |
| ECA_ResNet_vd-50 |  160 | TrivalAugment |  200  | 79.99 | 94.82 | 217.9 | [111](/configs/ImageNet/111.py) | [1](/configs/ImageNet/log/111.log) |

- Dropout 0.25, weight decay 4e-5
- References:
    - [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)
    - [PaddleClas](https://paddleclas.readthedocs.io/zh_CN/latest/advanced_tutorials/image_augmentation/ImageAugment.html)
