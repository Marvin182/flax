from jax import numpy as jnp

import tensorflow as tf


def pi_init(pi):
  """Wrapper to log-based weight initializer function.

  This initializer is used for the bias term in the classification subnet, as
  described in https://arxiv.org/abs/1708.02002

  Args:
    pi: the prior probability of detecting an object

  Returns:
    An array for initializing a module's weights / biases
  """

  def _inner(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * (-jnp.log((1 - pi) / pi))

  return _inner


def get_intersection(rect1, rect2):
  """Computes the intersection between two rectangles.

  Args:
    rect1: a list or tuple, which contains the coordinates of the top left and
           bottom right corners respectively: [x1, y1, x2, y2].
    rect2: a list or tuple, which contains the coordinates of the top left and
           bottom right corners respectively: [x1, y1, x2, y2].

  Returns:
    The area of the intersection of the two rectangles
  """
  overlap_x = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
  overlap_y = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
  return overlap_x * overlap_y


def jaccard_index(rect1, rect2):
  """Compute the Jaccard index of two rectangles.

  Args:
    rect1: a list or tuple, which contains the coordinates of the top left and
           bottom right corners respectively: [x1, y1, x2, y2].
    rect2: a list or tuple, which contains the coordinates of the top left and
           bottom right corners respectively: [x1, y1, x2, y2].

  Returns:
    The IoU of the two rectangles
  """
  # Get the intersection
  intersection = get_intersection(rect1, rect2)

  # Get the union value
  area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
  area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
  union = area1 + area2 - intersection

  # Return the IoU
  return intersection / union


@tf.function
def tf_jaccard_index(rects: tf.Tensor):
  rect1 = rects[0]
  rect2 = rects[1]

  # Compute the overlap on the X axis
  left_x = tf.math.maximum(rect1[0], rect2[0])
  right_x = tf.math.minimum(rect1[2], rect2[2])
  overlap_x = tf.math.maximum(0.0, right_x - left_x)

  # Compute the overlap on the Y axis
  lo_y = tf.math.maximum(rect1[1], rect2[1])
  hi_y = tf.math.minimum(rect1[3], rect2[3])
  overlap_y = tf.math.maximum(0.0, hi_y - lo_y)

  # Compute the area of the intersection
  intersection = overlap_x * overlap_y

  # Compute the area of the union
  area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
  area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
  union = area1 + area2 - intersection

  # Return the IoU
  return intersection / union


@tf.function
def tf_compute_regression_targets(rects: tf.Tensor,
                                  mean: tf.Tensor = None,
                                  std_dev: tf.Tensor = None):
  if mean is None:
    mean = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

  if std_dev is None:
    std_dev = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
