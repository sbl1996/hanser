import os
import random

import numpy as np

import tensorflow as tf

GLOBALS = {
    "seed": 0,
}


def set_seed(seed):
    GLOBALS['seed'] = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)