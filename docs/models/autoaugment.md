# Augmentation

## AutoAugment
| Model        | Size |  Augmentation | Epoch | Top 1 | Top 5 |  Time | Code |  Log  |
|--------------|:----:|:-------------:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|
| ResNet_vd-50 |  160 |  AutoAugment  |  200  | 78.99 | 94.33 | 232.4 | [98](/configs/ImageNet/98.py) | [1](/configs/ImageNet/log/98.log) [2](/configs/ImageNet/log/98-2.log) |
| ResNet_vd-50 |  160 |  RandAugment  |  200  | 78.89 | 94.08 | 218.1 | [99](/configs/ImageNet/99.py) | [1](/configs/ImageNet/log/99.log) [2](/configs/ImageNet/log/99-2.log) |
| ResNet_vd-50 |  160 |     RA-AA     |  200  | 79.01 | 94.35 | 249.1 | [100](/configs/ImageNet/100.py) | [1](/configs/ImageNet/log/100.log) [2](/configs/ImageNet/log/100-2.log) |
| ResNet_vd-50 |  160 | TrivalAugment |  200  | 79.26 | 94.17 | 213.1 | [109](/configs/ImageNet/109.py) | [1](/configs/ImageNet/log/109.log) [2](/configs/ImageNet/log/109-2.log) [3](/configs/ImageNet/log/109-3.log) |

- Reference papers:
    - [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
    - [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
    - [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://arxiv.org/abs/2103.10158)
- RA-AA randomly switches between AutoAugment and RandAugment for each sample.
- Note that TrivalAugment may not work well when using stronger models or combined with other data augmentation and regularization. See [Assemble](/docs/models/assemble) for more details.   