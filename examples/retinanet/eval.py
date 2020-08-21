from jax import numpy as jnp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import jax
import json

# A dictionary which converts model labels to COCO labels
MODEL_TO_COCO = {
    1: [1, "person"],
    2: [2, "bicycle"],
    3: [3, "car"],
    4: [4, "motorcycle"],
    5: [5, "airplane"],
    6: [6, "bus"],
    7: [7, "train"],
    8: [8, "truck"],
    9: [9, "boat"],
    10: [10, "traffic light"],
    11: [11, "fire hydrant"],
    12: [13, "stop sign"],
    13: [14, "parking meter"],
    14: [15, "bench"],
    15: [16, "bird"],
    16: [17, "cat"],
    17: [18, "dog"],
    18: [19, "horse"],
    19: [20, "sheep"],
    20: [21, "cow"],
    21: [22, "elephant"],
    22: [23, "bear"],
    23: [24, "zebra"],
    24: [25, "giraffe"],
    25: [27, "backpack"],
    26: [28, "umbrella"],
    27: [31, "handbag"],
    28: [32, "tie"],
    29: [33, "suitcase"],
    30: [34, "frisbee"],
    31: [35, "skis"],
    32: [36, "snowboard"],
    33: [37, "sports ball"],
    34: [38, "kite"],
    35: [39, "baseball bat"],
    36: [40, "baseball glove"],
    37: [41, "skateboard"],
    38: [42, "surfboard"],
    39: [43, "tennis racket"],
    40: [44, "bottle"],
    41: [46, "wine glass"],
    42: [47, "cup"],
    43: [48, "fork"],
    44: [49, "knife"],
    45: [50, "spoon"],
    46: [51, "bowl"],
    47: [52, "banana"],
    48: [53, "apple"],
    49: [54, "sandwich"],
    50: [55, "orange"],
    51: [56, "broccoli"],
    52: [57, "carrot"],
    53: [58, "hot dog"],
    54: [59, "pizza"],
    55: [60, "donut"],
    56: [61, "cake"],
    57: [62, "chair"],
    58: [63, "couch"],
    59: [64, "potted plant"],
    60: [65, "bed"],
    61: [67, "dining table"],
    62: [70, "toilet"],
    63: [72, "tv"],
    64: [73, "laptop"],
    65: [74, "mouse"],
    66: [75, "remote"],
    67: [76, "keyboard"],
    68: [77, "cell phone"],
    69: [78, "microwave"],
    70: [79, "oven"],
    71: [80, "toaster"],
    72: [81, "sink"],
    73: [82, "refrigerator"],
    74: [84, "book"],
    75: [85, "clock"],
    76: [86, "vase"],
    77: [87, "scissors"],
    78: [88, "teddy bear"],
    79: [89, "hair drier"],
    80: [90, "toothbrush"]
}


class CocoEvaluator():
  """We use the singleton pattern here to avoid re-reading the annotations.
  """

  def __init__(self, annotations_loc, remove_background=True, threshold=0.05):
    """Initializes a CocoEvaluator object.

    Args:
      annotations_loc: a path towards the .json files stroing the COCO/2014 
        ground truths for object detection. To get the annotations, please
        download the relevant files from https://cocodataset.org/#download
      remove_background: if True removes the anchors classified as background,
        i.e. having the greatest confidence on label 0
      threshold: a scalar which indicates the lower threshold (inclusive) for 
        the scores. Anything below this value will be set to -1.0

    """
    self.threshold = threshold
    self.remove_background = remove_background
    self.coco = COCO(annotations_loc)

  def extract_classifications(self, bboxes, scores):
    """Extracts the label for each bbox, and sorts the results by score.

    More specifically, after extracting each bbox's label, the bboxes and 
    scores are sorted in descending order based on score. The scores which fall
    below `threshold` are removed.

    Args:
      bboxes: a matrix of the shape (|B|, 4), where |B| is the number of 
        bboxes; each row contains the `[x1, y1, x2, y2]` of the bbox
      scores: a matrix of the shape (|B|, K), where `K` is the number of 
        classes in the object detection task

    Returns:
      A tuple consisting of the bboxes, a vector of length |B| containing 
      the label of each of the anchors, and a vector of length |B| containing 
      the label score. All elements are sorted in descending order relative
      to the score.  
    """
    # Extract the labels and max score for each anchor
    labels = np.argmax(scores, axis=1)

    # If requested, remove the anchors classified as background
    if self.remove_background:
      kept_idx = np.where(labels != 0)[0]
      labels = labels[kept_idx]
      bboxes = bboxes[kept_idx]
      scores = scores[kept_idx]

    # Get the score associated to each anchor's label
    scores = scores[np.arange(labels.shape[0]), labels]

    # Apply the threshold
    kept_idx = np.where(scores >= self.threshold)[0]
    scores = scores[kept_idx]
    labels = labels[kept_idx]
    bboxes = bboxes[kept_idx]

    # Sort everything in descending order and return
    sorted_idx = np.flip(np.argsort(scores, axis=0))
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    bboxes = bboxes[sorted_idx]

    return bboxes, labels, scores

  def __call__(self, bboxes, scores, img_id, scale):
    """Compute the COCO metrics.

    Args:
      bboxes: an array of the form (N, |B|, 4), where `N` is the batch size
        |B| is the number of bboxes containing the bboxes information
      scores: an array of the form (N, |B|, K), where `K` is the number of 
        classes. This array contains the confidence scores for each anchor
      img_od: an array of length `N`, containing the id of each of image
      scale: an array of length `N`, containing the scale of each image

    Returns:
      The COCO metrics as an array of length 12, defining the following:

      Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] 
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] 
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] 
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] 
    """

    def _inner(idx):
      # Get the sorted bboxes, labels and scores
      i_bboxes, i_labels, i_scores = self.extract_classifications(
          bboxes[idx, ...], scores[idx, ...])

      # Rescale bboxes to original size
      i_bboxes = i_bboxes / scale[idx]

      # Adjust bboxes to COCO standard: [x1, y1, w, h]
      i_bboxes[:, 2] -= i_bboxes[:, 0]
      i_bboxes[:, 3] -= i_bboxes[:, 1]

      # Iterate through the promising predictions, and pack them in json format
      i_img_id = img_id[idx]
      img_classifications = []
      for bbox, label, score in zip(i_bboxes, i_labels, i_scores):
        single_classification = {
            "image_id": i_img_id,
            "category_id": MODEL_TO_COCO[label][0],
            "bbox": bbox.tolist(),
            "score": score
        }
        img_classifications.append(single_classification)

      # Returned the classifications
      return img_classifications

    # Structure the predictions for each of the images and collect them
    detections = []
    for partial in map(_inner, range(bboxes.shape[0])):
      detections.extend(partial)

    # Create prediction object for producing mAP metric values
    print(detections)
    pred_object = self.coco.loadRes(detections)

    # Compute mAP on this specific image
    coco_eval = COCOeval(self.coco, pred_object, 'bbox')
    coco_eval.params.imgIds = img_id.flatten().tolist()  # Only this image
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
