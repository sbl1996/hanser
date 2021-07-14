## Data augmentation

Data augmentation is a common method of regularization in image classification tasks, particularly when the data is insufficient or the model is too large. In this section, we give a brief introduction and comparison of several image data augmentation methods.

<br/>

**Cutout**

```Cutout``` is a kind of dropout that occludes input image. It makes models more robust by (1) simulating the situation when the subject is partially occluded and (2) promoting the model to make full use of more content in the image.

examples?

<br/>


**RandomErasing**

Similar to ```Cutout```, ```RandomErasing``` aims to improve models' generalization ability on images with occlusions. Unlike ```Cutout```, ```RandomErasing``` is operated on images with a probability and requires pre-defined hyper-parameters.

examples?

<br/>


**GridMask**

```GridMask``` aims to achieve wide-range occlusion while maintaining information of target objects and context. ```GridMask``` assigns gird-arranged masks, instead of assigning a single rectangular one like methods above.

examples?

<br/>


The results of different methods on CIFAR10 dataset are shown as follows.

| Model       | Epoch | Augmentation | Acc    | Reference | Code                                                         |
| ----------- | ----- | ------------ | ------ | --------- | ------------------------------------------------------------ |
| WRN-28-10   | 300   | Basic        | 0.9663 | 0.9620    | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/25.py) |
| WRN-28-10   | 300   | RE-R         | 0.9727 | 0.9692    | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/26.py) |
| WRN-28-10   | 300   | GridMask     | -      | 0.9724    | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/26.py) |
| IResNet-110 | 300   | Basic        | 0.9605 | -         | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/22.py) |
| IResNet-110 | 300   | RE-M         | 0.9703 | -         | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/23.py) |
| IResNet-110 | 300   | RE-R         | 0.9711 | -         | [code](https://github.com/gourmets/experiments/blob/main/CIFAR10-TensorFlow-Yang/code/24.py) |

The logs of experiments above could be found [here](https://github.com/gourmets/experiments/tree/main/CIFAR10-TensorFlow-Yang/log) by experiment id.

