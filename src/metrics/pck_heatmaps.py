from .base_metric import BaseMetric
from .pck import PCK
from .utils import heatmaps_to_keypoints


class PCKHeatmaps(BaseMetric):
    """Percentage of Correct Keypoints (PCK) computed from heatmaps.

    This metric converts predicted heatmaps into keypoint coordinates and
    then evaluates them using the standard PCK metric.
    """

    def __init__(self, threshold=0.5):
        """Initialize the wrapped PCK metric.

        Args:
            threshold (float): Normalized distance threshold used to determine
                whether a keypoint prediction is correct.
        """
        self.metric = PCK(threshold)

    def update(self, preds, targets, norm_coefficient=25, **kwargs):
        """Update metric with a new batch of predictions.

        Args:
            preds (np.ndarray or torch.Tensor): Predicted heatmaps with shape
                (B, K, H, W), where:
                    B = batch size
                    K = number of keypoints
                    H, W = heatmap dimensions.
            targets (np.ndarray): Ground-truth keypoints with shape (B, K, D).
            norm_coefficient (float or np.ndarray): Normalization coefficient
                used for PCK threshold scaling.
            **kwargs: Additional unused arguments for compatibility.
        """

        # Convert predicted heatmaps to keypoint coordinates
        preds_kp = heatmaps_to_keypoints(preds)

        # Delegate metric computation to the base PCK metric
        self.metric.update(preds_kp, targets, norm_coefficient, **kwargs)

    def compute(self):
        """Compute the final PCK score.

        Returns:
            float: Percentage of correctly predicted keypoints.
        """
        return self.metric.compute()

    def reset(self):
        """Reset internal metric state."""
        self.metric.reset()