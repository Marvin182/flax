from jax import numpy as jnp

import jax
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


def non_max_suppression(bboxes, scores, t):
  """Implements the Non-Maximum Suppression algorithm.

  More specifically, this algorithm retains the bboxes based on their scores 
  (those that have a higher score are favored), and IoU's with the other bboxes
  (bboxes that have a high overlap with bboxes with higher scores are removed).

  Args:
    bboxes: a matrix of the form (|B|, 4), where |B| is the number of bboxes,
      and the columns represent the coordinates of each bbox: [x1, y1, x2, y2]
    scores: a vector of the form (|B|,) storing the confidence in each bbox
    t: the IoU threshold; overlap above this threshold with higher scoring 
      bboxes will imply the lower scoring bbox should be discarded

  Returns:
    The indexes of the bboxes which are retained after NMS is applied, as well
    as their indexes in the original matrix.
  """
  selected_idx = []

  # Split the bboxes so they're easier to manipulate throughout
  x1 = bboxes[:, 0]
  y1 = bboxes[:, 1]
  x2 = bboxes[:, 2]
  y2 = bboxes[:, 3]

  sorted_idx = jnp.argsort(scores)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)

  while sorted_idx.shape[0] > 0:
    # Select the index of the bbox with the highest score
    current = sorted_idx[-1]
    selected_idx.append(current)

    # Determine the height and width of the intersections with the current bbox
    xx1 = jnp.maximum(x1[current], x1[sorted_idx[:-1]])
    yy1 = jnp.maximum(y1[current], y1[sorted_idx[:-1]])
    xx2 = jnp.minimum(x2[current], x2[sorted_idx[:-1]])
    yy2 = jnp.minimum(y2[current], y2[sorted_idx[:-1]])

    width = jnp.maximum(0.0, xx2 - xx1 + 1)
    height = jnp.maximum(0.0, yy2 - yy1 + 1)

    # Compute the IoU between the current bbox and all the other bboxes
    intersection = width * height
    ious = intersection / (
        areas[current] + areas[sorted_idx[:-1]] - intersection)

    # Keep only the bboxes with the lower threshold
    sorted_idx = sorted_idx[jnp.where(ious < t)[0]]

  # Return the indexes of the non-suppressed bboxes
  selected_idx = jnp.array(selected_idx, dtype=jnp.int32)
  return bboxes[selected_idx, :], selected_idx


def vertical_pad(data, pad_count, dtype=jnp.float32):
  """Applies vertical padding to the data by adding extra rows with 0.
  
  Args:
    data: the data to be padded 
    pad_count: the number of extra rows of padding to be added to the data

  Returns:
    `data` with extra padding
  """
  pad_shape = (pad_count,) + data.shape[1:]
  pad_structure = jnp.zeros(pad_shape, dtype=dtype)
  return jnp.append(data, pad_structure, axis=0)


def top_k(scores, k, t=0.0):
  """Applies top k selection on the `scores` parameter.

  Args:
    scores: a vector of arbitrary length, containing non-negative scores, from 
      which only the at most top `k` highest scoring entries are selected.
    k: the maximal number of elements to be selected from `scores`
    t: a thresholding parameter (inclusive) which is applied on `scores`; 
      elements failing to meet the threshold are removed 

  Returns:
    Top top k entries from `scores` after thresholding with `t` is applied,
    as well as their indexes in the original vector.
  """
  idx = jnp.argsort(scores)[-k:]
  idx = idx[jnp.where(scores[idx] >= t)[0]]
  return scores[idx], idx


def filter_by_score(bboxes,
                    scores,
                    score_threshold: float = 0.05,
                    k: int = 1000,
                    per_class=False):
  """Apply top-k filtering on the bbox scores.

  More specifically, apply top `k` selection on the bboxes by filtering 
  out the bboxes which have a score lower than the `score_threshold`. The 
  filtering can be done either at the per class level or across all classes
  at the same time, as indicated by `per_class`.

  Args:
    bboxes: a matrix of the shape (|B|, 4), where |B| is the number of bboxes;
      each row will store the bbox's 4 coordinates: [x1, y1, x2, y2]   
    scores: a matrix of the shape (|B|, K), where K is the number of classes;
      each entry in a row will store the classification confidence of that class
    score_threshold: bboxes having a confidence score lower than this parameter
      will be discarded.
    k: the `k` parameter of the top k selection
    per_class: a flag which indicates whether the filtering and NMS operations 
      should be executed on a per class level    

  Returns:
    The bboxes retained after the filtering, and the indexes of these bboxes
    in the original data structure.
  """

  def _filter(inner_scores, labels):
    # First apply the top k filtering on the input
    top_k_scores, top_k_idx = top_k(inner_scores, k, score_threshold)
    top_k_bboxes = bboxes[top_k_idx, :]
    top_k_labels = labels[top_k_idx]
    return top_k_bboxes, top_k_scores, top_k_labels

  # Apply per class filtering
  if per_class:
    row_count = scores.shape[0]

    # Create some accumulators
    bbox_acc = jnp.zeros((0, 4))
    scores_acc = jnp.zeros(0)
    label_acc = jnp.zeros(0)

    # Iterate through all the classes, and apply filtering
    for i in range(scores.shape[1]):
      current_labels = jnp.ones(row_count, dtype=jnp.int32) * i
      current_scores = scores[:, i]
      temp_bboxes, temp_scores, temp_labels = _filter(current_scores,
                                                      current_labels)
      bbox_acc = jnp.append(bbox_acc, temp_bboxes, axis=0)
      scores_acc = jnp.append(scores_acc, temp_scores, axis=0)
      label_acc = jnp.append(label_acc, temp_labels, axis=0)

    return bbox_acc, scores_acc, label_acc
  else:
    current_labels = jnp.argmax(scores, axis=1)
    current_scores = scores[jnp.arange(scores.shape[0]), current_labels]
    return _filter(current_scores, current_labels)


def batch_filter_by_score(bboxes,
                          scores,
                          score_threshold: float = 0.05,
                          k: int = 1000,
                          per_class=False):
  """Apply top-k filtering on a batch. 

  For further explanation see `filter_by_score` documentation.

  Args:
    bboxes: an array of the shape (N, |B|, 4), where N is the batch count and 
      |B| is the number of bboxes; each row will store the bbox's 4 
      coordinates: [x1, y1, x2, y2]   
    scores: a matrix of the shape (N, |B|, K), where K is the number of classes;
      each entry in a row will store the classification confidence of that class
    score_threshold: bboxes having a confidence score lower than this parameter
      will be discarded.
    k: the `k` parameter of the top k selection. This value will also be used
      to pad the outputs if the number of valid bboxes is lower than `k`
    per_class: a flag which indicates whether the filtering and NMS operations 
      should be executed on a per class level; if True, then the output will 
      be padded to the size of `k x scores.shape[-1]`   

  Returns:
    A tuple containing: the number of true bboxes (i.e. not obtained through
    padding), the padded bboxes, the padded scores, the padded labels.  
  """
  pad_target = k if not per_class else k * scores.shape[-1]

  def _inner(idx):
    # Isolate the relevant image data and apply the filtering
    temp_bboxes = bboxes[idx, ...]
    temp_scores = scores[idx, ...]
    temp_bbox, temp_scores, temp_labels = filter_by_score(
        temp_bboxes, temp_scores, score_threshold, k, per_class)

    # Pad if necessary
    count = temp_bbox.shape[0]
    if count < pad_target:
      delta = pad_target - temp_bbox.shape[0]
      temp_bbox = vertical_pad(temp_bbox, delta)
      temp_scores = vertical_pad(temp_scores, delta)
      temp_labels = vertical_pad(temp_labels, delta)

    return count, temp_bbox, temp_scores, temp_labels

  # Create the structures which will store the filtered bboxes
  batch_size = bboxes.shape[0]
  counts = np.zeros((batch_size))
  filtered_bboxes = np.zeros((batch_size, pad_target, 4))
  filtered_scores = np.zeros((batch_size, pad_target))
  filtered_labels = np.zeros((batch_size, pad_target))

  # Apply the filtering function on each batch entry
  # FIXME: Find a way to do this in parallel 
  for idx, pack in enumerate(map(_inner, np.arange(batch_size))):
    counts[idx], filtered_bboxes[idx], filtered_scores[idx], filtered_labels[
        idx] = pack

  # Convert to jnp and return
  counts = jnp.array(counts)
  filtered_bboxes = jnp.array(filtered_bboxes)
  filtered_scores = jnp.array(filtered_scores)
  filtered_labels = jnp.array(filtered_labels)
  return counts, filtered_bboxes, filtered_scores, filtered_labels
