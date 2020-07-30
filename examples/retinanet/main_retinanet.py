from input_pipeline import read_data, prepare_data
from jax import numpy as jnp

import jax
import tensorflow as tf


def main(single_thread=True):
  # FIXME: The data is simply too big to fit in the memory of the GPU; need to 
  #        make it smaller somehow
  data = read_data()
  train_data, test_data = prepare_data(data, 2)
  
  if single_thread:
    import train_single_thread as logic
  else: 
    import train as logic
  
  # FIXME: Also do not forget to remove the first dimension in the batch on 
  #        single thread (can simply be done by commenting out the reshaping
  #        ops in prepare_split)
  logic.train_retinanet_model(None, None, (1000, 1000, 3), 
                              batch_size=2, try_restore=False)  


if __name__ == '__main__':
  # tf.config.experimental.set_visible_devices([], "GPU")
  main()
