import numpy as np
from .base_metric import BaseMetric
from .angles import CobbAngle
from .utils import heatmaps_to_keypoints


class CobbAngleHeatmaps(BaseMetric):
    """Cobb Angle computed from heatmap predictions.

    This metric converts predicted heatmaps into keypoint coordinates
    and then computes the Cobb angle between the specified cervical
    spine keypoints (C2 and C7).
    """

    def __init__(self):
        """Initialize the wrapped CobbAngle metric."""
        self.metric = CobbAngle()

    def update(self, preds, targets, orig_height, orig_width, predict_height=None, predict_width=None, c2_bl=0, c2_br=1, c7_bl=21, c7_br=22, **kwargs):
        """Update metric with a new batch of predictions.

        Args:
            preds (np.ndarray or torch.Tensor): Predicted heatmaps with shape
                (B, K, H, W), where:
                    B = batch size
                    K = number of keypoints
                    H, W = heatmap height and width.
            targets (np.ndarray): Ground-truth keypoints with shape (B, K, D).
            orig_height (int or list): Original height of the image(s).
            orig_width (int or list): Original width of the image(s).
            predict_height (int, optional): Height of the prediction. Defaults to None.
            predict_width (int, optional): Width of the prediction. Defaults to None.
            c2_bl (int, optional): Index for C2 bottom-left keypoint. Defaults to 0.
            c2_br (int, optional): Index for C2 bottom-right keypoint. Defaults to 1.
            c7_bl (int, optional): Index for C7 bottom-left keypoint. Defaults to 21.
            c7_br (int, optional): Index for C7 bottom-right keypoint. Defaults to 22.
            **kwargs: Additional unused arguments for compatibility.
        """
        preds_keypoints = heatmaps_to_keypoints(preds)

        _, heatmaps_height, heatmaps_width = preds[0].shape
        
        self.metric.update(preds_keypoints, targets, orig_height, orig_width, predict_height=heatmaps_height, predict_width=heatmaps_width, c2_bl=c2_bl, c2_br=c2_br, c7_bl=c7_bl, c7_br=c7_br)

    def compute(self):
        """Compute the final Cobb angle value.

        Returns:
            float: The computed Cobb angle.
        """
        return self.metric.compute()

    def reset(self):
        """Reset internal metric state."""
        self.metric.reset()