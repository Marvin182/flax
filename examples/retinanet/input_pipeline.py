from flax import jax_utils
from os import getenv

import itertools
import jax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


# These are the mean and std. dev of the COCO dataset
# TODO: compute the actual mean and std dev of the MS COCO dataset
COCO_MEAN = [0.485, 0.456, 0.406]
COCO_STD =  [0.229, 0.224, 0.225]


def preprocess_wrapper(func):
  """Which wraps `func`, such that the COCO data format is maintained.

  More specifically, it allows `func` to only implement logic for processing
  images, without having to deal with the logic of maintaining the dictionary
  structure implicitly produced by TFDS.

  Args:
    func: a function which takes a single unnamed parameter - the image -,
      processes it and then returns it.

  Returns:
    A function which can be passed a COCO data dictionary as produced by TFDS
  """
  def _inner(data):
    return {"image": func(data["image"]), "image/id": data["image/id"],
            "image/filename": data["image/filename"],
            "objects": data["objects"]}

  return _inner


def _standardize_image(image):
  """Standardizes the image values.

  Standardizes the image values to mean 0 with standard deviation 1. This
  function also normalizes the values prior to standardization.

  Args:
    image: the image to be standardized

  Returns:
    The standardized image
  """
  image = tf.image.convert_image_dtype(image, tf.float32)
  image -= tf.constant(COCO_MEAN, shape=[1, 1, 3])
  image /= tf.constant(COCO_STD, shape=[1, 1, 3])
  return image


def _resize_image(image, min_size=600, max_size=1000):
  """Resizes and pads the image to `max_size` x `max_size`.

  The image is resized, such that its shorter size becomes `min_size`, while
  maintaining the aspect ratio for the other side. If the greater side exceeds
  `max_size` after the initial resizing, the rescaling is done, such that the
  larger size is equal to `max_size`, which will mean that the shorter side
  will be less than the `min_size`. Finally, the image is padded, such that
  the final image will have dimensions of `max_size` x `max_size`. This approach
  has the advantage that the aspect ratio is maintained, and that the objects
  are not distorted.

  Args:
    image: the image to be resized
    min_size: the size of the shorter side, expressed in pixels
    max_size: the maximum size of the greater side, expressed in pixels

  Returns:
    The rescaled and padded image.
  """
  shape = tf.shape(image)
  short_side = tf.minimum(shape[0], shape[1])
  large_side = tf.maximum(shape[0], shape[1])

  # Create the constants
  two_constant = tf.constant(2, tf.int32)
  max_size_c_float = tf.constant(max_size, tf.float32)

  # Compute ratio such that image is not distorted, and is within bounds
  ratio = tf.constant(min_size, tf.float32) / tf.cast(short_side, tf.float32)
  if tf.math.less(max_size_c_float, tf.cast(large_side, tf.float32) * ratio):
    ratio = max_size_c_float / tf.cast(large_side, tf.float32)

  # Compute the new dimensions, and apply them to the image
  new_h = tf.cast(tf.cast(shape[0], tf.float32) * ratio, tf.int32)
  new_w = tf.cast(tf.cast(shape[1], tf.float32) * ratio, tf.int32)
  image = tf.image.resize(image, [new_h, new_w])

  # Apply uniform padding on the image
  max_size_c_int = tf.constant(max_size, tf.int32)
  offset_h = tf.cast((max_size_c_int - new_h) / two_constant, tf.int32)
  offset_w = tf.cast((max_size_c_int - new_w) / two_constant, tf.int32)
  return tf.image.pad_to_bounding_box(image, offset_h, offset_w, max_size,
                                      max_size)


def _standardize_resize(image, min_size=600, max_size=1000):
  """Applies a series of transformations to the input image.

  More specifically, this applies standardization and resizing. The output
  size of the image is `max_size` x `max_size`.

  Args:
    image: the image to be processed
    min_size: the size of the shorter side, expressed in pixels
    max_size: the maximum size of the greater side, expressed in pixels

  Returns:
    The standardized and resized image
  """
  image = _standardize_image(image)
  return _resize_image(image, min_size, max_size)


def _standardize_resize_flip(image, min_size=600, max_size=1000):
  """Applies a series of transformations to the input image.

  More specifically, this applies standardization, resizing, and then random
  horizontal image flipping. The output size of the image is `max_size` x
  `max_size`.

  Args:
    image: the image to be processed
    min_size: the size of the shorter side, expressed in pixels
    max_size: the maximum size of the greater side, expressed in pixels

  Returns:
    The standardized, resized, and (possibly) flipped image
  """
  image = _standardize_resize(image, min_size, max_size)
  return tf.image.random_flip_left_right(image)


def read_data(prng_seed: int = 0):
  """
  Reads the `COCO/2014` dataset and creates a `trainval35k` subset for
  training and uses the rest of the validation data for testing.

  Args:
     prng_seed: the seed to be used in the jax PRNG.

  Returns:
    A dictionary of the data, having the following shape:

    ```
    {
      "shape": (1000, 1000, 3),
      "train": {
          "count": 117783,
          "data": <train_data>
      },
      "test": {
          "count": 5504,
          "data": <test_data>
      }
    }
    ```

    The literals `117783` and `5504` are the size of the training and testing
    data respectively. As the COCO dataset size is constant, these can be
    computed and hardcoded beforehand.
  """
  # Values according to https://www.tensorflow.org/datasets/catalog/coco
  VAL_SIZE = 5504
  start = jax.random.randint(jax.random.PRNGKey(prng_seed), (1,), 0,
                             VAL_SIZE)[0]
  end = start + 35000

  # Read and prepare the data
  train_data, test_data = tfds.load('coco/2014',
                                    split=['train+validation[{}:{}]'.format(
                                      start, end),
                                      'validation[:{}]+validation[{}:]'
                                    .format(start, end)],
                                    data_dir=getenv("TFDS_DATA_DIR"))

  # TODO: Need to find a way to set image size easily, and also connect this
  #       to the image preprocessing
  data = {
    "shape": (1000, 1000, 3),
    "train": {
      "count": 117783,  # 117783 = 82783 + 35000
      "data": train_data
    },
    "test": {
      "count": VAL_SIZE,
      "data": test_data
    }
  }

  return data


def prepare_split(data, shape):
  device_count = jax.device_count()

  def _helper(batch):
    batch = batch._numpy()
    batch['image'] = tf.reshape(batch['image'], (device_count, -1) + shape)
    batch['image/id'] = tf.reshape(batch['image/id'], (device_count, -1, 1))
    batch['image/filename'] = tf.reshape(batch['image/filename'], (
      device_count, -1, 1))
    batch['objects'] = tf.reshape(batch['objects'], (device_count, -1, 1))
    return batch

  return jax_utils.prefetch_to_device(itertools.cycle(map(_helper, data)), 2)


def prepare_data(data, batch_size):
  """Process a COCO dataset, and produce training and testing input pipelines.

  Args:
    data: a dictionary of the form:
      ```
      {
        "shape": <shape>,
        "train": {
            "count": <train_example_count>,
            "data": <train_data>
        },
        "test": {
            "count": <test_example_count>,
            "data": <test_data>
        }
      }
      ```
    batch_size: the size of a batch

  Returns:
    A tuple containing a generator for the training batches and a generator for
    the testing batches respectively.
  """
  autotune = tf.data.experimental.AUTOTUNE

  # Create wrapped pre-processing methods
  standardize_resize = preprocess_wrapper(_standardize_resize)
  standardize_resize_flip = preprocess_wrapper(_standardize_resize_flip)

  # Prepare training data: standardize, resize and randomly flip the images
  train = data["train"]["data"].shuffle(1000).map(
    standardize_resize_flip, num_parallel_calls=autotune).batch(batch_size)
  train = prepare_split(train, data["shape"])

  # Prepare the test data: only standardize and resize
  test = data["test"]["data"].map(
    standardize_resize, num_parallel_calls=autotune).batch(batch_size),
  test = prepare_split(test, data["shape"])

  return train, test
