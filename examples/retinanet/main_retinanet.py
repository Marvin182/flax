from input_pipeline import read_data, prepare_data
from jax import numpy as jnp

import jax
import tensorflow as tf


def main(single_thread=True):
  data = read_data()
  train_data, test_data = prepare_data(data, 2, not single_thread)
  
  if single_thread:
    import train_single_thread as logic
  else: 
    import train as logic
  
  logic.train_retinanet_model(train_data, test_data, (1000, 1000, 3), 
                              batch_size=2, try_restore=False)  


if __name__ == '__main__':
  main()
