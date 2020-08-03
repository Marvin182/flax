from anchor import generate_all_anchors, AnchorConfig 
from flax import jax_utils
from jax import numpy as jnp
from os import getenv
from typing import Iterable, Tuple
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

  def __init__(self, min_size: int = 600, max_size: int = 1000, 
               foreground_t: float = 0.5, ignore_t: float = 0.4, 
               mean: Iterable[float] = None, std_dev: Iterable[float] = None, 
               bbox_mean: Iterable[float] = None, 
               bbox_std: Iterable[float] = None, label_shift: int = 0, 
               anchor_config : AnchorConfig = None):
    """Initializes a DataPreprocessor object which handles data preprocessing.

    Args:
      min_size: after resizing, the image should have the shorter side equal to 
        `min_size`; this may not happen when scaling towards `min_size` while 
        maintaining the aspect ratio implies that the longer size will exceed 
        `max_size`  
      max_size: the maximal allowed size for the longer side of the original 
        image after resizing; if the resized image exceeds `max_size`, it is 
        scaled down such that the longer size is equal to `max_size`
      foreground_t: a float in the range `[0, 1]` indicating a lower bound 
        (inclusive) for an anchor's maximal overlap with a ground truth 
        bounding box, such that the anchor is considered as foreground 
        and assigned a label and regression targets
      ignore_t: a float in the range `[0, foreground_t]` indicating the lower 
        bound (inclusive) of the maximal overlap of an anchor with a ground 
        truth bounding box. If the anchor's overlap is within the range 
        `[ignore_t, foreground_t)`, then the anchor is excluded from learning. 
        Note that when `ignore_t == foreground_t`, no anchor will be ignored.
      mean: a 3 element iterable, containing the means of the initial 3 image
        channels; this parameter is used for standardization
      std_dev: a 3 element iterable, containing the standard deviations of 
        the initial 3 image channels; this parameter is used for standardization
      bbox_mean: a 4 element iterable, containing the means of the 4 values 
        defining an anchor regression target; this parameter is used for 
        regression target standardization
      bbox_std: a 4 element iterable, containing the standard deviations of 
        the 4 values defining an anchor regression target; this parameter is 
        used for regression target standardization
      label_shift: a scalar used for shifting the ground truth labels; this 
        value is especially useful when trying to include token labels in the 
        data preprocessing pipeline, such as `0` for background 
      anchor_config: an AnchorConfig object, which contains the relevant
        information for statically unpacking the anchors in a fixed size image. 
    """
    assert min_size > 0 and max_size > 0 and min_size <= max_size, \
      "The following requirement is violated: 0 < min_size <= max_size"
    assert 1.0 >= foreground_t >= 0.0 and ignore_t >= 0.0 and \
      ignore_t <= foreground_t, "The following requirement is violated: " \
      "0.0 <= ignore_t <= foreground_t <= 1.0"

    self.min_size = min_size
    self.max_size = max_size
    self.foreground_t = foreground_t
    self.ignore_t = ignore_t
    self.label_shift = label_shift 

    # Create the mean and std deviation constants for regression standardization
    if bbox_mean is None:
      self.bbox_mean = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    else:
      self.bbox_mean = bbox_mean
    
    if bbox_std is None:
      self.bbox_std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
    else:
      self.bbox_std = bbox_std

    # If no mean and std deviation is provided, then reuse the ImageNet ones
    self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
    self.std = std_dev if std_dev is not None else [0.229, 0.224, 0.225] 

    # Generate all anchors for a `max_size` x `max_size` image
    if anchor_config is None:
      anchor_config = AnchorConfig()

    self.all_anchors = generate_all_anchors(
      (max_size, max_size), anchor_config.levels, anchor_config.strides, 
      anchor_config.sizes, anchor_config.ratios, anchor_config.scales, 
      clip=True)  # Note the clip here only clips agaist the padded image

    # Convert to tensor for the rest of the preprocessing
    self.all_anchors = tf.convert_to_tensor(self.all_anchors)

  @staticmethod
  def get_clipped_anchors(anchors: tf.Tensor, height: float, width: float):
    """Clips and returns the base anchors.

    More specifically, the x coordinates of the base anchors are clipped, 
    such that they are always found in the `[0, width]` interval, and 
    the `y` coordinates are always found in the `[0, height]` interval.

    Args:
      anchors: a tensor of the shape (|A|, 5) where the last column is reserved 
        for the anchor type
      height: the true height of the image
      width: the true width of the image

    Returns:
      A matrix of the form (|A|, 5), which contains the clipped anchors, as well
      as an extra column which can be used to store the status of the anchor.
    """
    # Clip the anchor coordinates
    x1 = tf.math.minimum(tf.math.maximum(anchors[:, 0], 0.0), width)
    y1 = tf.math.minimum(tf.math.maximum(anchors[:, 1], 0.0), height)
    x2 = tf.math.minimum(tf.math.maximum(anchors[:, 2], 0.0), width)
    y2 = tf.math.minimum(tf.math.maximum(anchors[:, 3], 0.0), height)

    return tf.stack([x1, y1, x2, y2, anchors[:, 4]], axis=1)

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

  def resize_image(self, image):
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
      lambda x: tf.convert_to_tensor([x[0], 1.0 - x[3], x[2], 1.0 - x[1]], 
        dtype=tf.float32),
      bboxes, fn_output_signature=tf.TensorSpec(4, dtype=tf.float32))
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
    """Get the indexes of the anchors inside and outside the `shape` rectangle.

    Args:
      anchors: a matrix of anchors with shape (-1, 4)
      shape: a list or tuple with the shape [height, width], which defines 
        the rectangle having coordinates: (0, 0) and (width, height)
  
    Returns:
      Two column vectors, with the indexes into `anchors` of the anchors which
      fall inside `shape`, and those which do not, respecitvely 
    """
    two = tf.constant(2.0, dtype=tf.float32)
    mid_x = (anchors[:, 0] + anchors[:, 2]) / two
    mid_y = (anchors[:, 1] + anchors[:, 3]) / two
    centers = tf.transpose(tf.stack([mid_x, mid_y]))
    
    # We are guaranteed to have the centers >= 0: check only upper bounds 
    in_anchors = tf.math.logical_and(centers[:, 0] < shape[1], 
                                     centers[:, 1] < shape[0])
    out_anchors = tf.math.logical_not(in_anchors)

    # Get the indexes of the inner and outer anchors
    row_idxs = tf.range(tf.shape(anchors)[0])
    in_idx = tf.boolean_mask(row_idxs, in_anchors)  
    out_idx = tf.boolean_mask(row_idxs, out_anchors)  

    return in_idx, out_idx 

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
    overlaps = tf.map_fn(tf_jaccard_index, (anchors, bboxes), 
                         fn_output_signature=tf.float32)
    return tf.reshape(overlaps, [anchor_count, bbox_count])

  def compute_foreground_ignored(self, overlaps : tf.Tensor):
    """Identifies the row indices of the foreground and ignored anchors.

    More specifically, this method will inspect the argmax of `overlaps` 
    on eachrow, and computes the membership to the foreground and ignored 
    anchor sets based on the overlap of the argmax. 

    Args:
      overlaps: an (|A|, |B|) matrix, where each entry stores the IoU of an 
        anchor and a bbox. Here, |A| is the count of anchors, and |B| is 
        the count of ground truth bboxes

    Returns:
      The indices of the foreground and ignored anchors respectively, as well
      as the argmax indices in the `overlaps`. 
    """
    # Get the max overlap value for each of the anchors
    row_idxs = tf.range(tf.shape(overlaps)[0], dtype=tf.int32)
    argmax = tf.argmax(overlaps, axis=1, output_type=tf.int32)
    indexes = tf.stack([row_idxs, argmax], axis=1)
    max_overlaps = tf.gather_nd(overlaps, indexes)  
    
    # Get the indexes of the foreground anchors
    lower = tf.math.greater_equal(max_overlaps, self.foreground_t)
    foreground_idx = tf.boolean_mask(row_idxs, lower)

    # Get the indexes of the ignored anchors
    lower = tf.math.greater_equal(max_overlaps, self.ignore_t)
    upper = tf.math.less(max_overlaps, self.foreground_t)
    in_range = tf.math.logical_and(lower, upper)
    ignored_idx = tf.boolean_mask(row_idxs, in_range)

    return foreground_idx, ignored_idx, argmax

  def compute_regression_targets(self, anchors,  bbox):
    """Computes the regression targets of the `anchors`.

    This method also applies standardization on the regression targets. 
    The `anchors` and `bbox` parameters must have the same number of rows. 
    That is, for an anchor at row `i` in `anchors`, bbox must store its ground 
    truth at row `i` in `bbox`. 

    Args:
      anchors: the anchors for which the targets are computed, having a 
        shape like (|A|, 4)
      bbox: the bboxes for which the anchor targets are computed, having a 
        shape like (|A|, 4)

    Returns:
      The standardized anchor targets, having the shape (|A|, 4).
    """
    # Get the heights and widths
    heights = anchors[:, 3] - anchors[:, 1]
    widths = anchors[:, 2] - anchors[:, 0] 

    # Compute the regression targets
    dx1 = (bbox[:, 0] - anchors[:, 0]) / widths
    dy1 = (bbox[:, 1] - anchors[:, 1]) / heights
    dx2 = (bbox[:, 2] - anchors[:, 2]) / widths
    dy2 = (bbox[:, 3] - anchors[:, 3]) / heights

    # Standardize the targets, and return
    targets = tf.transpose(tf.stack([dx1, dy1, dx2, dy2]))
    return  (targets - self.bbox_mean) / self.bbox_std

  def compute_anchors_and_labels(self, bboxes, labels, height, width):
    """Computes the anchors, their type, as well as their targets.

    More specifically, this method will compute all the anchors within the 
    padded image, and will determine whether they are foreground, background
    or ignored. The method also determines the targets for both regression 
    and classification of the candidate anchors, based on the IoU with the
    available ground truth bounding boxes. 
    
    Args:
      bboxes: a tensor for the shape `(|B|, 4)`, which holds the ground truth
        bounding boxes of the image
      labels: a tensor of length `|B|`, storing the label of each ground truth
        bounding box
      height: the height of the true image within the padded image
      width: the width of the true image within the padded image

    Returns:
      A triple, which stores the following elements:
        * `anchors`: an (|A|, 5) tensor, where |A| is equal to the number of 
          anchors that can be deployed in the padded image. The first 4 elements
          on each row represent the [x1, y1, x2, y2] coordinates of the anchor,
          while the last entry can either be -1 (ignored), 0 (background) or 
          1 (foreground).
        * `classification_labels`: an (|A|,) shaped tensor, which contains the 
          classification target for each anchor. `0` is considered background.
          During training, only the non-ignored anchors should be used.
        * `regression_targets`: an (|A|, 4) shaped tensor, which contains the 
          regression targets for each of the anchor's 4 coordinates. During 
          training, only the foreground anchors should be considered.
    """
    # Copy to avoid recomputing
    anchors = tf.identity(self.all_anchors)

    # Find the inner anchors, clip them, and find their overlaps
    in_idx, out_idx = self.filter_outer_anchors(anchors, [height, width])
    in_anchors = tf.gather(anchors, in_idx)
    in_anchors = self.get_clipped_anchors(in_anchors, height, width)
    anchors = tf.tensor_scatter_nd_update(anchors, tf.expand_dims(in_idx, 1), 
                                          in_anchors)

    overlaps = self.compute_anchor_overlaps(in_anchors, bboxes)
    foreground_idx, ignored_idx, argmax = self.compute_foreground_ignored(
      overlaps)

    # Set to 1 the last column of those anchors which are foreground
    extra_coord = 4 * tf.ones(tf.shape(foreground_idx)[0], dtype=tf.int32)
    foreground_idx = tf.stack([foreground_idx, extra_coord], axis=1)
    in_anchors = tf.tensor_scatter_nd_update(
      in_anchors, foreground_idx, tf.ones(
        tf.shape(foreground_idx)[0], dtype=tf.float32))

    # Set to -1 the last column of those anchors which are ignored
    extra_coord = 4 * tf.ones(tf.shape(ignored_idx)[0], dtype=tf.int32)
    ignored_idx = tf.stack([ignored_idx, extra_coord], axis=1)
    in_anchors = tf.tensor_scatter_nd_update(
      in_anchors, ignored_idx, tf.ones(
        tf.shape(ignored_idx)[0], dtype=tf.float32) * -1.0)

    # Update the foreground / ignore labels in the original anchors structure
    extra_coord = 4 * tf.ones(tf.shape(in_idx)[0], dtype=tf.int32)
    in_idx = tf.stack([in_idx, extra_coord], axis=1)
    anchors = tf.tensor_scatter_nd_update(anchors, in_idx, in_anchors[:, -1])
    
    # Update the out-of-bounds anchors in the original anchors structure
    extra_coord = 4 * tf.ones(tf.shape(out_idx)[0], dtype=tf.int32)
    out_idx = tf.stack([out_idx, extra_coord], axis=1)
    anchors = tf.tensor_scatter_nd_update(anchors, out_idx, tf.ones(
        tf.shape(out_idx)[0], dtype=tf.float32) * -1.0)

    # Compute the classification targets
    label_idx = tf.expand_dims(argmax, axis=1)
    classification_labels = tf.gather_nd(labels, label_idx)
    indices = tf.expand_dims(in_idx[:, 0], 1)
    classification_labels = tf.scatter_nd(indices, classification_labels, 
      (tf.shape(anchors)[0],))

    # Prepare the regression target computation
    foreground_anchors = tf.gather(anchors, foreground_idx[:, 0])
    argmax_foreground = tf.gather(argmax, foreground_idx[:, 0])
    foreground_bboxes = tf.gather(bboxes, argmax_foreground)

    # Compute the regression targets
    foreground_regression_targets = self.compute_regression_targets(
      foreground_anchors, foreground_bboxes)

    # Gradually adjust the regression targets to the right dimensions
    indices = tf.expand_dims(foreground_idx[:, 0], 1)
    temp_targets = tf.scatter_nd(indices, foreground_regression_targets, 
      (tf.shape(in_idx)[0], 4))

    indices = tf.expand_dims(in_idx[:, 0], 1)
    regression_targets = tf.scatter_nd(indices, temp_targets,
      (tf.shape(anchors)[0], 4))

    # Return the computed results
    return anchors, classification_labels, regression_targets

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
      labels = data["objects"]["label"] + self.label_shift
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
      anchors, classification_labels, regression_targets = \
        self.compute_anchors_and_labels(bboxes, labels, new_image_h, 
                                        new_image_w)

      # Return the preprocessed batch
      return {
        "image": new_image, 
        "size": [tf.cast(new_image_h, tf.int32), tf.cast(new_image_w, tf.int32), 
                new_image_c],
        "anchor_type": tf.cast(anchors[:, -1], dtype=tf.int32), 
        "classification_labels": classification_labels,
        "regression_targets": regression_targets
      }
    return _inner


def is_annotated(data):
  """Predicate which identifies images with annotations.

  Args:
    data: a data instance in the context of object detection as provided by
      TFDS
  
  Returns:
    True if the instance has annotations, False otherwise. 
  """
  return tf.math.greater(tf.size(data["objects"]["label"]), 0)


def read_data(rng):
  """
  Reads the `COCO/2014` dataset and creates a `trainval35k` subset for
  training and uses the rest of the validation data for testing.

  Args:
     rng: JAX PRNGKey.

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
  start = jax.random.randint(rng, (1,), 0, VAL_SIZE)[0]
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


def prepare_data(
    data: tf.data.Dataset,
    per_device_batch_size: int,
    distributed_training: bool = True,
    shape: Iterable[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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
            "data": <test_data> } } ```
    per_device_batch_size: The batch size per device (GPU or TPU core). E.g.
      per_device_batch_size = global_batch_size // jax.device_count().
    distributed_training: True if the data is prepare for distributed training,
      and hence will require an additional dimension for the number of devices
    shape: an iterable data structure of two elements: min side size, and
      max size side

  Returns:
    A tuple containing the preprocessed datasets for training and testing.
  """
  if not shape:
    shape = (600, 1000)

  # Note: we shift the labels by 1 since 0 is reserved for background
  batch_preprocessor = DataPreprocessor(min_size=shape[0], max_size=shape[1], 
                                        label_shift=1)
  autotune = tf.data.experimental.AUTOTUNE

  # Define the relevant leading dimensions for the batch
  if distributed_training:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [per_device_batch_size]

  # Prepare training data: standardize, resize and randomly flip the images
  train = data["train"]["data"].filter(is_annotated).repeat().shuffle(
      per_device_batch_size * 16, seed=0).map(
          batch_preprocessor(augment_image=True), num_parallel_calls=autotune)
  for batch_size in reversed(batch_dims):
    train = train.batch(batch_size, drop_remainder=True)

  # Prepare the test data: only standardize and resize
  test = data["test"]["data"].filter(is_annotated).map(
      batch_preprocessor(), num_parallel_calls=autotune)
  for batch_size in reversed(batch_dims):
    test = test.batch(batch_size, drop_remainder=True)

  return train, test
