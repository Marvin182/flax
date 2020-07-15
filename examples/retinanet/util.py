from dataclasses import dataclass, field
from jax import numpy as jnp


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


@dataclass
class AnchorConfig:
  """This class contains the necessary information for unpacking anchors.

  It should be mentioned that the length of `sizes` and `strides` must be equal
  to the number of layers in RetinaNet's head. `scales` and `ratios` are lists
  of arbitrary lengths.
  """
  sizes : list = field(default_factory=lambda: [32, 64, 128, 256, 512])
  strides : list = field(default_factory=lambda: [8, 16, 32, 64, 128])
  ratios : list = field(default_factory=lambda: [0.5, 1, 2])
  scales : list = field(default_factory=lambda: [1.0, 2.0 ** (1.0 / 3.0),
                                                 2.0 ** (2.0 / 3.0)])
