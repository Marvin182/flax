from jax import numpy as jnp

import tensorflow as tf

CATEGORY_MAP = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush"
}


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
