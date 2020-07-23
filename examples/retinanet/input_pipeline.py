from anchor import generate_all_anchors, AnchorConfig 
from flax import jax_utils
from jax import numpy as jnp
from os import getenv
from util import tf_jaccard_index

import itertools
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

# This controls the maximal number of bbox annotations in an image
MAX_PADDING_ROWS = 100


class DataPreprocessor:
  """This class handles data preprocessing for object detection tasks.
  """

  def __init__(self, min_size=600, max_size=1000, mean=None, std_dev=None, 
               anchor_config : AnchorConfig = None):
    assert min_size > 0 and max_size > 0 and min_size <= max_size, \
      "The following requirement is violated: 0 < min_size <= max_size"

    self.min_size = min_size
    self.max_size = max_size

    # If no mean and std deviation is provided, then reuse the ImageNet ones
    self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
    self.std = std_dev if std_dev is not None else [0.229, 0.224, 0.225] 

    # Generate all anchors for a `max_size` x `max_size` image
    if anchor_config is None:
      anchor_config = AnchorConfig()

    self.all_anchors = generate_all_anchors(
      (max_size, max_size), anchor_config.levels, anchor_config.strides, 
      anchor_config.sizes, anchor_config.ratios, anchor_config.scales)
    
    # Convert to tensor for the rest of the preprocessing
    self.all_anchors = tf.convert_to_tensor(self.all_anchors)

  def standardize_image(self, image):
    """Standardizes the image values.

    Standardizes the image values to mean 0 with standard deviation 1. This
    function also normalizes the values prior to standardization.

    Args:
      image: the image to be standardized

    Returns:
      The standardized image
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= tf.constant(self.mean, shape=[1, 1, 3])
    image /= tf.constant(self.std, shape=[1, 1, 3])
    return image

  def resize_image(self, image, min_size=600, max_size=1000):
    """Resizes and pads the image to `max_size` x `max_size`.

    The image is resized, such that its shorter size becomes `min_size`, while
    maintaining the aspect ratio for the other side. If the greater side exceeds
    `max_size` after the initial resizing, the rescaling is done, such that the
    larger size is equal to `max_size`, which will mean that the shorter side
    will be less than the `min_size`. Finally, the image is padded on the right
    and lower side, such that the final image will be `max_size` x `max_size`.

    Args:
      image: the image to be resized

    Returns:
      The rescaled and padded image, together with the scaling ratio
    """
    shape = tf.shape(image)
    short_side = tf.minimum(shape[0], shape[1])
    large_side = tf.maximum(shape[0], shape[1])

    # Create the constants
    max_size_c_float = tf.constant(self.max_size, tf.float32)

    # Compute ratio such that image is not distorted, and is within bounds
    ratio = tf.constant(self.min_size, tf.float32) / tf.cast(short_side, 
                                                             tf.float32)
    if tf.math.less(max_size_c_float, tf.cast(large_side, tf.float32) * ratio):
      ratio = max_size_c_float / tf.cast(large_side, tf.float32)

    # Compute the new dimensions, and apply them to the image
    new_h = tf.cast(tf.cast(shape[0], tf.float32) * ratio, tf.int32)
    new_w = tf.cast(tf.cast(shape[1], tf.float32) * ratio, tf.int32)
    image = tf.image.resize(image, [new_h, new_w])

    # Apply uniform padding on the image
    return tf.image.pad_to_bounding_box(
      image, 0, 0, self.max_size, self.max_size), ratio

  @staticmethod
  def augment_image(image, bboxes):
    """This applies data augmentation on the image and its associated bboxes.

    Currently, the image only applies horizontal flipping to the image and 
    its ground truth bboxes. 

    Args:
      image: the image to be augmented
      bboxes: a 2D tensor, which stores the coordinates of the ground truth 
        bounding boxes 

    Returns:
      A tuple consisting of the transformed image and bboxes.
    """
    ## BBoxes should be normalized, and have the structure: [y1, x1, y2, x2]
    image = tf.image.flip_left_right(image)
    bboxes = tf.map_fn(
      lambda x: tf.convert_to_tensor([x[0], 1.0 - x[1], x[2], 1.0 - x[3]], 
      dtype=tf.float32), bboxes)
    return image, bboxes

  @staticmethod
  def pad(data, output_rows, dtype=tf.float32):
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

  @staticmethod
  def filter_outer_anchors(anchors : tf.Tensor, shape : tf.Tensor) -> tf.Tensor:
    """Get the indexes of the anchors inside the `shape` rectangle.

    Args:
      anchors: a matrix of anchors with shape (-1, 4)
      shape: a list or tuple with the shape [height, width], which defines 
        the rectangle having coordinates: (0, 0) and (width, height) respectively
  
    Returns:
      A column vector, with True in the positions where the anchor lies within 
      the rectangle, and False otherwise
    """
    two = tf.constant(2.0, dtype=tf.float32)
    mid_x = (anchors[:, 0] + anchors[:, 2]) / two
    mid_y = (anchors[:, 1] + anchors[:, 3]) / two
    centers = tf.transpose(tf.stack([mid_x, mid_y]))
    
    # We are guaranteed to have the centers >= 0: check only upper bounds 
    in_anchors = tf.math.logical_and(centers[:, 0] < shape[1], 
                                     centers[:, 1] < shape[0])
    return tf.boolean_mask(tf.range(0, tf.shape(anchors)[0]), in_anchors)  
     

  @staticmethod
  def compute_anchor_overlaps(anchors : tf.Tensor, bboxes : tf.Tensor):
    """Computes the IoU of each anchor against each bbox.

    Given an (|A|, 4) matrix for the anchors, and a (|B|, 4) matrix for the 
    bboxes, this method will output an (|A|, |B|) matrix, where each entry
    (i, j) stores the IoU of anchor `i` and bbox `j`. 

    Args:
      anchors: an (|A|, 4) matrix for the anchors.
      bboxes: a (|B|, 4) matrix for the bboxes.

    Returns:
      An (|A|, |B|) matrix, where each entry stores the IoU of an anchor and 
      a bbox.
    """
    bbox_count = tf.shape(bboxes)[0]
    anchor_count = tf.shape(anchors)[0]

    # Compute the Cartesian Product between the anchors and the bboxes 
    repeats = tf.ones(anchor_count, dtype=tf.int32) * bbox_count
    anchors = tf.repeat(anchors, repeats, axis=0)
    bboxes = tf.tile(bboxes, [anchor_count, 1])
    
    # Compute the overlaps
    overlaps = tf.map_fn(tf_jaccard_index, (anchors, bboxes), dtype=tf.float32)
    return tf.reshape(overlaps, [anchor_count, bbox_count])

  @staticmethod
  def compute_labels(overlaps, foreground_threshold=0.5, ignore_threshold=0.4):
    # Get the max overlap value for each of the anchors
    row_idxs = tf.range(0, tf.shape(overlaps)[0], dtype=tf.int64)
    argmax = tf.argmax(overlaps, axis=1)
    indexes = tf.stack([row_idxs, argmax], axis=1)
    max_overlaps = tf.gather_nd(overlaps, indexes)
    
    # Get the indexes of the foreground anchors
    lower = tf.math.greater_equal(max_overlaps, foreground_threshold)
    foreground = tf.boolean_mask(row_idxs, lower)

    # Get the indexes of the ignored anchors
    lower = tf.math.greater_equal(max_overlaps, ignore_threshold)
    upper = tf.math.less(max_overlaps, foreground_threshold)
    in_range = tf.math.logical_and(lower, upper)
    ignored = tf.boolean_mask(row_idxs, in_range) 

    return foreground, ignored

  def __call__(self, augment_image=False, augment_probability=0.5):
    """Creates a TF compatible function which can be used to preprocess batches. 

    The generated function will unpack an object detection dataset, as returned
    by TFDS, preprocess it by standardizing and rescaling the image to 
    a given size, while also maintaining its aspect ratio using padding. If 
    the `augment_image` is `True`, then the image will be augmented with 
    `augment_probability` likelihood. The initial data, is expected to be a 
    TFDS FeaturesDict, which has minimally the following structure:

    ```
      {
      'image': Image(shape=(None, None, 3), dtype=tf.uint8),
      'objects': Sequence({
          'area': tf.int64,
          'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
          'id': tf.int64,
          'is_crowd': tf.bool,
          'label': ClassLabel(shape=(), dtype=tf.int64),
      }
    ```

    Args:
      augment_image: a flag which indicates whether to randomly augment 
        the image and its associated anchors
      augment_probability: the probability with which to perform augmentation
        if `augment_image` is True

    Returns:
      [TODO]: explain what the structure of the output will be like
    """
    assert 0.0 <= augment_probability <= 1.0, "augment_probability must be " \
                                              "in the range [0.0, 1.0]"

    def _inner(data):
      # Unpack the dataset elements
      image = data["image"]
      is_crowd = data["objects"]["is_crowd"]
      labels = data["objects"]["label"]
      bboxes = data["objects"]["bbox"]
      bbox_count = tf.shape(bboxes)[0]

      if augment_image and tf.random.uniform(
        [], minval=0.0, maxval=1.0) >= augment_probability:
        image, bboxes = self.augment_image(image, bboxes)

      # Preprocess the image, and compute the size of the new image
      image = self.standardize_image(image)
      new_image, ratio = self.resize_image(image)
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

      # Perform anchor specific operations: filtering, IoU, and labelling
      anchors = tf.identity(self.all_anchors)
      in_idx = self.filter_outer_anchors(anchors, [new_image_h, new_image_w])
      in_anchors = tf.gather(anchors, in_idx)
      overlaps = self.compute_anchor_overlaps(in_anchors, bboxes)
      foreground, ignored = self.compute_labels(overlaps)
      
      # Pad the bboxes, to make TF batching possible
      is_crowd = self.pad(is_crowd, MAX_PADDING_ROWS, dtype=tf.bool)
      labels = self.pad(labels, MAX_PADDING_ROWS, dtype=tf.int64)
      bboxes = self.pad(bboxes, MAX_PADDING_ROWS)

      # Return the preprocessed batch
      return {
        "image": new_image, 
        "size": [tf.cast(new_image_h, tf.int32), tf.cast(new_image_w, tf.int32), 
                new_image_c],
        "bbox_count": bbox_count,
        "is_crowd": is_crowd,
        "labels": labels,
        "bbox": bboxes,
        "overlaps": overlaps,
        "foreground": foreground,
        "ignored": ignored
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

    # FIXME: This might need to be removed, since it's just for debug 
    print("Original ignored shape:", batch['ignored'].shape)
    print("Original foreground shape:", batch['foreground'].shape)
    batch['overlaps'] = jnp.reshape(
      batch['overlaps'], (device_count, -1, batch['overlaps'].shape[-2], batch['overlaps'].shape[-1]))    
    batch['ignored'] = jnp.reshape(
      batch['ignored'], (device_count, -1, batch['ignored'].shape[0]))
    batch['foreground'] = jnp.reshape(
      batch['foreground'], (device_count, -1, batch['foreground'].shape[0]))
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
  batch_preprocessor = DataPreprocessor()

  # Prepare training data: standardize, resize and randomly flip the images
  train = data["train"]["data"].repeat().shuffle(batch_size * 16, seed=0).map(
    batch_preprocessor(augment_image=True), num_parallel_calls=autotune)
  train = prepare_split(train.batch(batch_size), data["shape"])

  # Prepare the test data: only standardize and resize
  test = data["test"]["data"].map(batch_preprocessor(), 
                                  num_parallel_calls=autotune)
  test = prepare_split(test.batch(batch_size), data["shape"])

  return train, test
