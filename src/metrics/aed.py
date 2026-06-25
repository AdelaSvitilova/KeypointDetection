import numpy as np
from .base_metric import BaseMetric
from .utils import scale_keypoints

class AED(BaseMetric):
    """Average Euclidean Distance (AED) metric.

    This metric computes the mean Euclidean distance between predicted
    keypoints and ground-truth keypoints.

    The metric accumulates distances across batches and computes the final
    average when `compute()` is called.
    """

    def __init__(self):
        """Initialize metric accumulators."""
        self.sum_dist = 0.0
        self.total_points = 0

    def update(self, preds, targets, orig_height=None, orig_width=None, predict_height=None, predict_width=None, **kwargs):
        """Update metric with a new batch of predictions.

        Args:
            preds (np.ndarray): Predicted keypoints with shape (B, K, D),
                where B is batch size, K is number of keypoints,
                and D >= 2 (x, y, ...).
            targets (np.ndarray): Ground-truth keypoints with the same shape.
            **kwargs: Additional unused arguments for compatibility.
        """

        # Extract x, y coordinates only
        preds_xy = preds[..., :2]
        targets_xy = targets[..., :2]

        if orig_height is not None and predict_height is not None and orig_width is not None and predict_width is not None:
            if not (np.array_equal(orig_height, predict_height) and np.array_equal(orig_width, predict_width)):
                preds_xy = scale_keypoints(preds_xy, [orig_height, orig_width], [predict_height, predict_width])
                targets_xy = scale_keypoints(targets_xy, [orig_height, orig_width], [predict_height, predict_width])

        # Compute Euclidean distance per keypoint
        # Shape: (B, K)
        dist = np.sqrt(np.sum((preds_xy - targets_xy) ** 2, axis=2))

        # Accumulate total distance and number of evaluated points
        self.sum_dist += np.sum(dist)
        self.total_points += dist.size

    def compute(self):
        """Compute the final average Euclidean distance.

        Returns:
            float: Mean Euclidean distance across all processed keypoints.
        """
        return self.sum_dist / self.total_points if self.total_points > 0 else 0.0

    def reset(self):
        """Reset metric state."""
        self.sum_dist = 0.0
        self.total_points = 0