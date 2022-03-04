import tensorflow as tf
import numpy as np
import graph_tool.all as gt

import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='test', shuffle_files=True)
print(ds.numpy())
