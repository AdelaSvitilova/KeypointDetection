from .base_metric import BaseMetric
from .aed import AED
from .utils import heatmaps_to_keypoints


class AEDHeatmaps(BaseMetric):
    """Average Euclidean Distance computed from heatmap predictions.

    This metric converts predicted heatmaps into keypoint coordinates
    and then computes the Average Euclidean Distance (AED) between the
    predicted and ground-truth keypoints.
    """

    def __init__(self):
        """Initialize the wrapped AED metric."""
        self.metric = AED()

    def update(self, preds, targets, **kwargs):
        """Update metric with a new batch of predictions.

        Args:
            preds (np.ndarray or torch.Tensor): Predicted heatmaps with shape
                (B, K, H, W), where:
                    B = batch size
                    K = number of keypoints
                    H, W = heatmap height and width.
            targets (np.ndarray): Ground-truth keypoints with shape (B, K, D).
            **kwargs: Additional unused arguments for compatibility.
        """

        # Convert predicted heatmaps into keypoint coordinates
        preds_kp = heatmaps_to_keypoints(preds)

        # Delegate distance computation to the AED metric
        self.metric.update(preds_kp, targets)

    def compute(self):
        """Compute the final AED value.

        Returns:
            float: Average Euclidean distance between predicted
            and ground-truth keypoints.
        """
        return self.metric.compute()

    def reset(self):
        """Reset internal metric state."""
        self.metric.reset()