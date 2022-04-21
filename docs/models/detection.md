# Detection

## RetinaNet
| Backbone  | Size |   Augmentation   | BS  | Epoch |  AP   |            Code             |                                                              Log                                                              |
|-----------|:----:|:----------------:|:---:|:-----:|:-----:|:---------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|
| ResNet-50 | 1024 |     Standard     | 32  |  24   | 36.02 |  [60](/configs/COCO/60.py)  |                                 [1](/configs/COCO/log/60.log) [2](/configs/COCO/log/60-2.log)                                 |
| ResNet-50 | 896  |   Pad (random)   | 16  |  12   | 36.49 | [100](/configs/COCO/100.py) |               [1](/configs/COCO/log/100.log) [2](/configs/COCO/log/100-2.log) [3](/configs/COCO/log/100-3.log)                |
| ResNet-50 | 1024 |   Pad (random)   | 32  |  24   | 37.75 |  [98](/configs/COCO/98.py)  | [1](/configs/COCO/log/98.log) [2](/configs/COCO/log/98-2.log) [3](/configs/COCO/log/98-3.log) [4](/configs/COCO/log/98-4.log) |
| ResNet-50 | 1024 |   R(0.5, 2.0)    | 32  |  24   | 39.55 |  [65](/configs/COCO/65.py)  |                                                 [1](/configs/COCO/log/65.log)                                                 |
| ResNet-50 | 1024 |   R(0.5, 2.0)    | 64  |  50   | 41.06 |  [79](/configs/COCO/79.py)  |                                                 [1](/configs/COCO/log/79.log)                                                 |

- All above are trained with horizontal flip augmentation.
- Pad (random) is a new way to pad the input image which reduces overfitting. Only with it, we could achieve similar AP as mmdetection, becuase mmdetection actually pads the input images randomly.


## GFLV2
| Backbone  | Size |     Augmentation      | BS  | Optimizer | Epoch |  AP   |            Code             |                                                                Log                                                                |
|-----------|:----:|:---------------------:|:---:|:---------:|:-----:|:-----:|:---------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
| ResNet-50 | 1024 |      R(0.5, 2.0)      | 64  |    SGD    |  50   | 44.68 |  [76](/configs/COCO/76.py)  |                                   [1](/configs/COCO/log/76.log) [2](/configs/COCO/log/76-2.log)                                   |
| ResNet-50 | 1024 |      R(0.5, 2.0)      | 64  |   AdamW   |  50   | 45.11 |  [93](/configs/COCO/93.py)  |   [1](/configs/COCO/log/93.log) [2](/configs/COCO/log/93-2.log) [3](/configs/COCO/log/93-3.log) [4](/configs/COCO/log/93-4.log)   |
| ResNet-50 | 1024 |      R(0.1, 2.0)      | 64  |   AdamW   |  50   | 45.39 |  [97](/configs/COCO/97.py)  |                   [1](/configs/COCO/log/97.log) [2](/configs/COCO/log/97-2.log) [3](/configs/COCO/log/97-3.log)                   |
| ResNet-50 | 1024 |   R(0.1, 2.0) + EMA   | 64  |   AdamW   |  50   | 45.68 | [107](/configs/COCO/107.py) | [1](/configs/COCO/log/107.log) [2](/configs/COCO/log/107-2.log) [3](/configs/COCO/log/107-3.log) [4](/configs/COCO/log/107-4.log) |

- [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://arxiv.org/abs/2011.12885)