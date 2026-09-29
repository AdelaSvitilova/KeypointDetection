import numpy as np
from .base_metric import BaseMetric
from .utils import heatmaps_to_keypoints, scale_keypoints


class CobbAngle(BaseMetric):
    """Cobb Angle error metric computed from keypoint coordinates.

    This metric calculates the Cobb angle between two lines defined by four
    keypoints (typically the endplates of C2 and C7 vertebrae) for both
    predictions and ground-truth targets, and tracks the mean absolute
    angular error.
    """

    def __init__(self):
        """Initialize the CobbAngle metric state."""
        self.error_sum = 0.0
        self.total_points = 0

    def _compute_cobb(self, p1, p2, q1, q2):
        """Compute the angle (in degrees) between two lines defined by four points.

        Args:
            p1 (np.ndarray): First point of the first line with shape (B, 2).
            p2 (np.ndarray): Second point of the first line with shape (B, 2).
            q1 (np.ndarray): First point of the second line with shape (B, 2).
            q2 (np.ndarray): Second point of the second line with shape (B, 2).

        Returns:
            np.ndarray: Computed angles in degrees with shape (B,).
        """
        v1 = np.stack([p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1]], axis=1)
        v2 = np.stack([q2[:, 0] - q1[:, 0], q2[:, 1] - q1[:, 1]], axis=1)
        dot = np.sum(v1 * v2, axis=1)
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0] 
        return np.degrees(np.atan2(det, dot))

    def update(self, preds, targets, orig_height, orig_width, predict_height=None, predict_width=None, c2_bl=0, c2_br=1, c7_bl=21, c7_br=22, **kwargs):
        """Update metric state with a new batch of predictions and targets.

        If original and prediction image dimensions are provided and differ,
        the keypoints are automatically scaled to the original dimensions before
        computing the angle.

        Args:
            preds (np.ndarray): Predicted keypoints with shape (B, K, 2).
            targets (np.ndarray): Ground-truth keypoints with shape (B, K, 2).
            orig_height (int or list or np.ndarray): Original height of the image(s).
            orig_width (int or list or np.ndarray): Original width of the image(s).
            predict_height (int, optional): Height of the prediction. Defaults to None.
            predict_width (int, optional): Width of the prediction. Defaults to None.
            c2_bl (int, optional): Index for C2 bottom-left keypoint. Defaults to 0.
            c2_br (int, optional): Index for C2 bottom-right keypoint. Defaults to 1.
            c7_bl (int, optional): Index for C7 bottom-left keypoint. Defaults to 21.
            c7_br (int, optional): Index for C7 bottom-right keypoint. Defaults to 22.
            **kwargs: Additional unused arguments for compatibility.
        """
        if orig_height is not None and predict_height is not None and orig_width is not None and predict_width is not None:
            if not (np.array_equal(orig_height, predict_height) and np.array_equal(orig_width, predict_width)):
                preds = scale_keypoints(preds, [orig_height, orig_width], [predict_height, predict_width])
                targets = scale_keypoints(targets, [orig_height, orig_width], [predict_height, predict_width])

        cobb_preds = self._compute_cobb(
            preds[:, c2_bl, :], 
            preds[:, c2_br, :], 
            preds[:, c7_bl, :], 
            preds[:, c7_br, :]
        )
        cobb_targets = self._compute_cobb(
            targets[:, c2_bl, :], 
            targets[:, c2_br, :], 
            targets[:, c7_bl, :], 
            targets[:, c7_br, :]
        )

        error = (cobb_preds - cobb_targets + 180) % 360 - 180

        self.error_sum += np.mean(abs(error))
        self.total_points += 1
        

    def compute(self):
        """Compute the final mean absolute Cobb angle error.

        Returns:
            float: Average absolute angular error in degrees.
        """
        return self.error_sum / self.total_points if self.total_points > 0 else 0.0

    def reset(self):
        """Reset internal metric state."""
        self.error_sum = 0.0
        self.total_points = 0