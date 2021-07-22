# Attention

## RetinaNet
| Backbone  | Size | Augmentation | Loss | BS | Epoch |   AP  | Time | Code |
|-----------|:----:|:------------:|:----:|:--:|:-----:|:-----:|:----:|:----:|
| ResNet-50 |  640 |   Standard   |  L1  | 32 |   24  | 37.19 |      |   5  |



| Backbone  | Size | Augmentation | Loss | BS | Epoch |   AP  | Time | Code | Log |
|-----------|:----:|:------------:|:----:|:--:|:-----:|:-----:|:----:|:----:|:---:|
| ResNet-50 |  640 |   Standard   |  L1  | 32 |   24  | 37.19 |      | [5](/configs/COCO-Detection/5.py) | [5](/configs/COCO-Detection/5.log) |


## ECA-Net
| Model             | Size | Augmentation | Epoch | Top 1 | Top 5 |   Time   |  Code  |  Log  |
|-------------------|:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|:-----:|
| ResNet_vd-50      |  160 |   Standard   |  120  | 79.24 | 94.67 |   203.1  |   [24](/configs/ImageNet/24.py)   | [1](/configs/ImageNet/log/24.log) |
| RegNetY-1.6GF (C) |  160 |   Standard   |  100  | 77.27 | 93.46 |   160.9  |   [35](/configs/ImageNet/35.py)   | [1](/configs/ImageNet/log/35.log) |
| RegNetY-1.6GF (F) |  160 |   Standard   |  100  | 77.55 | 93.83 |   159.7  |   [38](/configs/ImageNet/38.py)   | [1](/configs/ImageNet/log/38.log) |

- [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)
- ECA-Net seems to not work well in RegNet, no matter put it at the middle conv (C) or at the final of block (F).


## Reference
| Model         | Size | Augmentation | Epoch | Top 1 | Top 5 |   Time   |  Code  |  Log  |
|---------------|:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|:-----:|
| ResNet_vd-50  |  160 |   Standard   |  120  | 78.54 | 94.33 |   192.1  |   [50](/configs/ImageNet/50.py)   | [1](/configs/ImageNet/log/50.log) [2](/configs/ImageNet/log/50-2.log) [3](/configs/ImageNet/log/50-3.log) |
| RegNetY-1.6GF |  160 |   Standard   |  100  | 77.86 | 94.01 |   161.8  |   [32](/configs/ImageNet/32.py)   | [1](/configs/ImageNet/log/32.log) |