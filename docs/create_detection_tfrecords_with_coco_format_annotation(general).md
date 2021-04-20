# Create Detection TFRecords with COCO Format Annotation

<font size="4" color='gray'>Use SVHN Detection Dataset as an example</font>

[comment]: <> (reproted date: 2021.4.17)

## Important Lib Versions
+ hanser: head **56aa07a**
+ numpy: 1.19.5
+ tensorflow: 2.3.0
+ pycocotools: 2.0.2
+ tensorflow-probability: 0.11.0
+ tensorflow-addons: 0.12.1
+ tensorflow-datasets: 4.2.0


## How to create TFRecords
1. **(Optional)** Generate COCO format annotation

    + Download <font color='CadetBlue'>**SVHN Detection Dataset**</font> from [here](http://ufldl.stanford.edu/housenumbers/)
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
   + Run the annotation format transform script and generate *.json* files for trainset and testset respectively

2. Organize dataset and annotation
   + Rename annotation *.json* files of trainset <font color='CadetBlue'>(, valset)</font> and testset to *train.json* <font color='CadetBlue'>(,*val.json*)</font> and *test.json*, respectively
   + Move dataset and annotation files to *~/tensorflow_datasets/downloads/manual/* (default path for tfds), 
     or you can change the path to <font color='CadetBlue'>*YOUR_DATASET_PATH*</font> and set *manual_dir* later
   + Create a path structure like this
     ```
     |-train
       |-1.png
       ...
     |-test
       |-1.png
       ...
     |-val (Optional)
       |-1.png
       ...
     |-train.json
     |-test.json
     |-val.json (Optional)
     ```
3. Generate TFRecords
   + Generate a file *dataset.py* that defines the dataset (change the class name and parameters to <font color='CadetBlue'>**YOUR OWN DATASET**</font>)
    
     - *dataset.py* looks like this:
       ```
       from hanser.datasets.detection.general import CocoBuilder

       class Svhn(CocoBuilder):
           NUM_CLASSES = 10            # Define number of label classes 
           LABEL_OFFSET = 1            # Shift the label index start with LABEL_OFFSET to 0
           SPLITS = ['train', 'test']  # Define dataset splits (choices: 'train', 'val' and 'test')
       ```
     - file *dataset.py* defines an inheritance of definition of 
         **Tensorflow Dataset** with COCO format annotation
   
   + Run the command below if dataset is in default path
     ```
     tfds build dataset.py
     ```
   + Change the command if use <font color='CadetBlue'>*YOUR_DATASET_PATH*</font>
     ```
     tfds build --manual_dir=YOUR_DATASET_PATH dataset.py
     ```
   + Find TFRecords files in *~/tensorflow_datasets/<font color='CadetBlue'>svhn</font>* by default
     - name of the folder is determined by customized class name in *dataset.py*
     - if use <font color='CadetBlue'>*YOUR_TARGET_PATH*</font>, use command below:
       ```
       tfds build [--manual_dir=YOUR_DATASET_PATH] --data_dir=YOUR_TARGET_PATH dataset.py
       ```
     
## References     
- see reference of [Using Command Line to Generate TFRecords](https://www.tensorflow.org/datasets/cli)
- see reference of [Writing Custom Datasets](https://www.tensorflow.org/datasets/add_dataset) and 
     [Official COCO TensorFlow Dataset](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/coco.py)
     to get more infomation about **Tensorflow Dataset**
