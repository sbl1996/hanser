# Create Object Detection TFRecords with COCO Format Annotation

<font size="4" color='gray'>Use SVHN Detection Dataset as an example</font>

[comment]: <> (reproted date: 2021.4.17)

## Important Lib Versions
+ hanser: head **6a7486f**
+ numpy: 1.19.5
+ tensorflow: 2.3.0
+ pycocotools: 2.0.2
+ tensorflow-probability: 0.11.0
+ tensorflow-addons: 0.12.1
+ tensorflow-datasets: 4.2.0


## How to create TFRecords
1. **(Optional)** Generate COCO format annotation

    + Download <font color='AntiqueWhite'>**SVHN Detection Dataset**</font> from [here](http://ufldl.stanford.edu/housenumbers/)
    + Extract dataset and create a path structure (eg: like below)

    ```
    |-(annotation format transform script)
    |-data
      |-train
        |-1.png
        ...
        |-digitStruct.mat (SVHN original annotation)
      |-test
        |-1.png
        ...
        |-digitStruct.mat (SVHN original annotation)
    ```

2. Move dataset and annotation
   + Rename annotation *.json* files of train set and test set to *train.json* and *test.json* respectively
   + Move dataset and annotation to *~/tensorflow_datasets/downloads/manual/*
   + Create a path structure like this
    ```
    |-train
      |-1.png
      ...
    |-test
      |-1.png
      ...
    |-train.json
    |-test.json
    ```
3. Generate TFRecords
    + Change variable *ann_file* in *examples/det/atss_svhn.py* to full path of *train.json*
    + Run the script to check the correctness of the TFRecords
    ```
    python examples/det/atss_svhn.py
    ```
    + Find TFRecords files in *~/tensorflow_datasets/<font color='AntiqueWhite'>svhn</font>* 
      - name of the folder is determined by customized <font color='AntiqueWhite'>**TensorFlow Dataset**</font> (definition see *hanser/datasets/tfds/detection/svhn.py*)
      - see reference of [Writing Custom Datasets](https://www.tensorflow.org/datasets/add_dataset) and 
        [Official COCO to TFRecords](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py)
        to get more infomation about <font color='AntiqueWhite'>**Tensorflow Dataset**</font>