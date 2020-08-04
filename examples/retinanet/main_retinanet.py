from input_pipeline import read_data, prepare_data
from jax import numpy as jnp
from spicy import magic
from configs.default import get_config

import jax
import tensorflow as tf


WORK_DIR = "./distributed_retinanet"


def main(single_thread: bool = True):
  # Some training parameters
  batch_size = 2

  # Read the data
  rng = jax.random.PRNGKey(0)
  rng, input_key = jax.random.split(rng)
  data = read_data(input_key)

  # Process the images
  img_size = (224, 224, 3)
  train_data, test_data = prepare_data(data, batch_size, not single_thread,
                                       img_size[:2])

  if single_thread:
    import train_single_thread as logic
    logic.train_retinanet_model(rng,
                                train_data,
                                test_data,
                                img_size,
                                batch_size=batch_size,
                                try_restore=False)
  else:
    import train as logic
    config = get_config()
    logic.train_and_evaluate(config, WORK_DIR)


if __name__ == '__main__':
  # Disable GPU usage (if any)
  tf.config.experimental.set_visible_devices([], "GPU")

  # Set up the training process
  # magic()
  main(single_thread=False)
