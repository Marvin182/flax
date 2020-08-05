from absl import app, logging
from input_pipeline import read_data, prepare_data
from jax import numpy as jnp
from spicy import magic
from configs.default import get_default_config

import jax
import tensorflow as tf
import sys

WORK_DIR = "./distributed_retinanet"


def run_model(single_device: bool = True):
  # Execute the training based on single or multi-device options
  if single_device:
    import train_single_thread as logic
    # Read the data
    rng = jax.random.PRNGKey(0)
    rng, input_key = jax.random.split(rng)
    data = read_data(input_key)

    batch_size = 2
    img_size = (224, 224, 3)
    train_data, test_data = prepare_data(data, batch_size, not single_device,
                                         img_size[:2])
    logic.train_retinanet_model(
        rng,
        train_data,
        test_data,
        img_size,
        batch_size=batch_size,
        try_restore=False)
  else:
    import train as logic
    config = get_default_config()
    logic.train_and_evaluate(config, WORK_DIR)


def main(argv):
  # Delete superfluous argument
  del argv

  # Disable GPU usage (if any)
  tf.config.experimental.set_visible_devices([], "GPU")

  # Set up the training process
  jax.config.enable_omnistaging()
  magic()
  run_model(single_device=False)


if __name__ == '__main__':
  app.run(main)
