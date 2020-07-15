from train import create_model

import jax
import tensorflow.compat.v2 as tf

def main():
  # Prepare the TF environment
  tf.enable_v2_behavior()
  # tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPUs, if any

  # Create the model, debug only
  model = create_model(jax.random.PRNGKey(0))


if __name__ == '__main__':
  main()
