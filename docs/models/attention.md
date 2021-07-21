# Attention

## SENet
| Model           | Size | Augmentation | Epoch | Top 1 | Top 5 | Time (s) | Script |
|-----------------|:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|
| ResNet_vd-50 |  160 |   Standard   |  120  | 78.93 | 94.48 |   208.1  |   [23](/configs/ImageNet/23.py)   |
| ResNet_vd-50 |  224 |   Standard   |  120  | 78.86 | 94.28 |   377.2  |   [101](/configs/ImageNet/101.py)  |


## ECA-Net
| Model             | Size | Augmentation | Epoch | Top 1 | Top 5 | Time (s) | Script |
|-------------------|:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|
| ResNet_vd-50  |  160 |   Standard   |  120  | 79.24 | 94.67 |   203.1  |   24   |
| RegNetY-1.6GF (C) |  160 |   Standard   |  100  | 77.27 | 93.46 |   160.9  |   35   |
| RegNetY-1.6GF (F) |  160 |   Standard   |  100  | 77.55 | 93.83 |   159.7  |   38   |

Notes:
- ECA-Net seems to not work well in RegNet, no matter put it at the middle conv or at the final of block.


## Reference
| Model         | Size | Augmentation | Epoch | Top 1 | Top 5 | Time (s) | Script |
|---------------|:----:|:------------:|:-----:|:-----:|:-----:|:--------:|:------:|
| ResNet_vd-50  |  160 |   Standard   |  120  | 78.54 | 94.33 |   192.1  |   50   |
| RegNetY-1.6GF |  160 |   Standard   |  100  | 77.86 | 94.01 |   161.8  |   32   |