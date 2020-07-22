from flax import jax_utils
from jax import numpy as jnp
from os import getenv

import itertools
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

# These are constants relevant for the data preprocessing stage 
MAX_PADDING_ROWS = 100

# These are the mean and std. dev of the COCO dataset
# TODO: compute the actual mean and std dev of the MS COCO dataset
COCO_MEAN = [0.485, 0.456, 0.406]
COCO_STD =  [0.229, 0.224, 0.225]


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
  will be less than the `min_size`. Finally, the image is padded on the right
  and lower side, such that the final image will be `max_size` x `max_size`.

  Args:
    image: the image to be resized
    min_size: the size of the shorter side, expressed in pixels
    max_size: the maximum size of the greater side, expressed in pixels

  Returns:
    The rescaled and padded image, together with the scaling ratio
  """
  shape = tf.shape(image)
  short_side = tf.minimum(shape[0], shape[1])
  large_side = tf.maximum(shape[0], shape[1])

  # Create the constants
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
  return tf.image.pad_to_bounding_box(image, 0, 0, max_size, max_size), ratio


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


def preprocess_wrapper(func, flip_image=False):
  """Wraps `func`, such that the COCO data format is maintained.

  More specifically, it allows `func` to only implement logic for processing
  images, without having to deal with the logic of unpacking or rescaling 
  the bboxes. 

  Args:
    func: a function which takes a single unnamed parameter - the image -,
      processes it and returns it, and the new scale of the image.
    flip_image: a flag which indicates whether to randomly flip the image and
      associated anchors

  Returns:
    A function which can be passed a COCO data dictionary as produced by TFDS
  """
  def _pad(data, output_rows, dtype=tf.float32):
    """Adds extra rows to the data. 

    Args:
      data: a 1D or 2D dataset to be padded
      output_rows: a scalar indicating the number of rows of the padded data
      dtype: the TF dtype used for padding

    Returns:
      The padded data
    """
    data_shape = tf.shape(data)
    to_pad = tf.math.maximum(0, output_rows - data_shape[0])
    padding_shape = [to_pad, data_shape[1]] if len(data_shape) == 2 else to_pad
    padding = tf.zeros(padding_shape, dtype=dtype)
    return tf.concat([data, padding], axis=0)

  def _augment(image, bboxes):
    ## BBoxes should be normalized, and have the structure: [y1, x1, y2, x2]
    image = tf.image.flip_left_right(image)
    bboxes = tf.map_fn(
      lambda x: tf.convert_to_tensor([x[0], 1.0 - x[1], x[2], 1.0 - x[3]], 
      dtype=tf.float32), bboxes)
    return image, bboxes

  def _inner(data):
    # Unpack the dataset elements
    image = data["image"]
    is_crowd = data["objects"]["is_crowd"]
    labels = data["objects"]["label"]
    bboxes = data["objects"]["bbox"]
    bbox_count = tf.shape(bboxes)[0]

    if flip_image and tf.random.uniform([], minval=0.0, maxval=1.0) >= 0.5:
        image, bboxes = _augment(image, bboxes)

    # Preprocess the image, and compute the size of the new image
    new_image, ratio = func(image)
    original_image_size = tf.cast(tf.shape(image), tf.float32)
    new_image_h = original_image_size[0] * ratio
    new_image_w = original_image_size[1] * ratio
    new_image_c = tf.cast(original_image_size[2], tf.int32)

    # Invert the x's and y's, to make access more intuitive
    y1 = bboxes[:, 0] * new_image_h
    x1 = bboxes[:, 1] * new_image_w
    y2 = bboxes[:, 2] * new_image_h
    x2 = bboxes[:, 3] * new_image_w
    bboxes = tf.stack([x1, y1, x2, y2], axis=1)
    
    # Pad the bboxes, to make TF batching possible
    is_crowd = _pad(is_crowd, MAX_PADDING_ROWS, dtype=tf.bool)
    labels = _pad(labels, MAX_PADDING_ROWS, dtype=tf.int64)
    bboxes = _pad(bboxes, MAX_PADDING_ROWS)

    return {
      "image": new_image, 
      "size": [tf.cast(new_image_h, tf.int32), tf.cast(new_image_w, tf.int32), 
               new_image_c],
      "bbox_count": bbox_count,
      "is_crowd": is_crowd,
      "labels": labels,
      "bbox": bboxes
    }

  return _inner


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
    # Convert the dataset to np array
    batch = tfds.as_numpy(batch)

    # Reshape the data such that it can be distributed to the devices 
    batch['image'] = jnp.reshape(batch['image'], (device_count, -1) + shape)
    batch['size'] = jnp.reshape(batch['size'], (device_count, -1, 3))
    batch['bbox_count'] = jnp.reshape(batch['bbox_count'], 
                                      (device_count, -1, 1))
    batch['is_crowd'] = jnp.reshape(batch['is_crowd'], 
                                    (device_count, -1, MAX_PADDING_ROWS))
    batch['labels'] = jnp.reshape(batch['labels'], 
                                  (device_count, -1, MAX_PADDING_ROWS))
    batch['bbox'] = jnp.reshape(batch['bbox'], 
                                (device_count, -1, MAX_PADDING_ROWS, 4))
    return batch

  return map(_helper, data)


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
  standardize_resize_flip = preprocess_wrapper(_standardize_resize, 
                                               flip_image=True)

  # Prepare training data: standardize, resize and randomly flip the images
  train = data["train"]["data"].repeat().shuffle(batch_size * 16, seed=1).map(
    standardize_resize_flip, num_parallel_calls=autotune)
  train = prepare_split(train.batch(batch_size), data["shape"])

  # Prepare the test data: only standardize and resize
  test = data["test"]["data"].map(standardize_resize, 
                                  num_parallel_calls=autotune)
  test = prepare_split(test.batch(batch_size), data["shape"])

  return train, test
