# hanser

Hanser is a library to help with training for different tasks in TensorFlow.

- [Sponsors](#sponsors)
- [Features](#features)
- [Models](#models)
- [Examples](#examples)

## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Features

- Simplified training and evalution, running on CPU, GPU and **TPU**, maybe more powerful than Keras
- Easily extended to supervised learning tasks (Classification, Detection, Segmentation, NAS, Super Resolution, Keypoint Detection)
- Reproduction of popular computer vision models, modules and tricks **with experimental results**

## Models
- [ResNet](/docs/models/resnet.md)
- [Attention](/docs/models/attention.md)
- [AutoAugment](/docs/models/autoaugment.md)
- [Assemble](/docs/models/assemble.md)
- [Detection](/docs/models/detection.md)

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