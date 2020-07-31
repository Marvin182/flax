from input_pipeline import read_data, prepare_data
from jax import numpy as jnp

import jax
import tensorflow as tf


def disable_gpu():
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(single_thread: bool = True):
  img_size = (224, 224, 3)
  data = read_data()
  train_data, test_data = prepare_data(data, 2, not single_thread, img_size[:2])
  
  if single_thread:
    import train_single_thread as logic
  else: 
    import train as logic
  
  logic.train_retinanet_model(train_data, test_data, img_size, 
                              batch_size=2, try_restore=False)  


if __name__ == '__main__':
  disable_gpu()
  main()
