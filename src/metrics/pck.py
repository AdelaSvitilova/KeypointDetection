import numpy as np
from .base_metric import BaseMetric


class PCK(BaseMetric):
    """Percentage of Correct Keypoints (PCK).

    A keypoint prediction is considered correct if the Euclidean distance
    between the predicted and ground-truth keypoint is below a threshold
    multiplied by a normalization coefficient.

    The metric accumulates results across batches and computes the final
    percentage when `compute()` is called.
    """

    def __init__(self, threshold=0.5):
        """Initialize the PCK metric.

        Args:
            threshold (float): Normalized distance threshold that determines
                whether a keypoint is considered correct.
        """
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, preds, targets, norm_coefficient=25, **kwargs):
        """Update metric with a new batch.

        Args:
            preds (np.ndarray): Predicted keypoints with shape (B, K, D),
                where B is batch size, K is number of keypoints,
                and D >= 2 (x, y, ...).
            targets (np.ndarray): Ground-truth keypoints with the same shape.
            norm_coefficient (float or np.ndarray): Normalization coefficient
                used to scale the distance threshold. Can be:
                - scalar (same for all samples)
                - shape (B,) for per-sample normalization.
            **kwargs: Additional unused arguments for compatibility.
        """

        # Extract x and y coordinates
        preds_xy = preds[..., :2]
        targets_xy = targets[..., :2]

        # Compute Euclidean distance for each keypoint
        # Shape: (B, K)
        dist = np.sqrt(np.sum((preds_xy - targets_xy) ** 2, axis=2))

        # If normalization is per-sample (shape B), expand to (B, 1)
        if np.ndim(norm_coefficient) == 1:
            norm_coefficient = norm_coefficient[:, None]

        # Count correct predictions based on the normalized threshold
        self.correct += np.sum(dist < self.threshold * norm_coefficient)
        self.total += dist.size

    def compute(self):
        """Compute the final PCK score.

        Returns:
            float: Percentage of correctly predicted keypoints.
        """
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0