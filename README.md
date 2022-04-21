# hanser

`hanser` is a library to help with training for different tasks in TensorFlow 2.X.

- [Sponsors](#sponsors)
- [Features](#features)
- [Models](#models)
- [Highlights](#highlights)
- [Examples](#examples)

## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Features

- Simplified training and evalution, running on CPU, GPU and **TPU**, maybe more powerful than Keras
- Easily extended to supervised learning tasks (Classification, Detection, Segmentation, NAS, Super Resolution, Keypoint Detection)
- Reproduction of popular computer vision models, modules and tricks **with experimental results**

## Models
- [ResNets series](/docs/models/resnet.md)
- [Attention](/docs/models/attention.md)
- [AutoAugment](/docs/models/autoaugment.md)
- [Assemble](/docs/models/assemble.md)
- [Detection](/docs/models/detection.md)

## Highlights
### [InplaceABN](/examples/highlights/inplace_abn.py)
InplaceABN is an improved variant of gradient checkpointing which drastically reduce the training memory with negligible computation cost. We make it available on **TPU** and have carefully tested the correctness and performance. With InplaceABN, the maximal batch size may increase 20%-50% with a tiny time overhead (<10%).

### [EMA](/examples/official/cls/imagenet/mobilenetv3.py)
EMA is a simple and efficient way to stabilize training and improve performance. We integrate it into `hanser` and make it very easy to use. It has been carefully tested on our reproduced [MobileNetV3](/examples/official/cls/imagenet/mobilenetv3.py). 

### [Data Augmentation](/hanser/transform)
`hanser` provides out-of-the-box reimplementation of various data augmentation and regularization (CutMix, ResizeMix, RandAugment, TrivialAugment, DropBlock, ...). All of them are well tested and work seamlessly with **TPU**. 

## Examples

### Image Classification
- [CIFAR](/examples/official/cls/cifar)
  - [Wide-ResNet-28-10](/examples/official/cls/cifar/wrn/WRN-28-10.py)
  - [PyramidNet + ShakeDrop](/examples/official/cls/cifar/shakedrop/PyramidNet-270-a200.py)
  - [CutMix](/examples/official/cls/cifar/cutmix/PyramidNet.py)
- [ImageNet](/examples/official/cls/imagenet)
  - [ResNet-50-D](/examples/official/cls/imagenet/resnet_vd50.py)
  - [MobileNetV3](/examples/official/cls/imagenet/mobilenetv3.py)

### Object Detection
- [VOC](/examples/official/det/voc)
  - [FCOS](/examples/official/det/voc/fcos.py)
- [COCO](/examples/official/det/coco)
  - [GFLV2](/examples/official/det/coco/gflv2.py)

### Semantic Segmentation
- [Cityscapes](/examples/official/seg/cityscapes)
  - [DeepLabV3+](/examples/official/seg/cityscapes/deeplabv3p.py)

### Hyperparameter Tuning
- [Optuna](/examples/hpo/optuna)
  - [MNIST](/examples/hpo/optuna/mnist.py)
- [Microsoft NNI](/examples/hpo/nni)
  - [MNIST](/examples/hpo/nni/mnist.py)
  - [CIFAR100](/examples/hpo/nni/wrn.py)