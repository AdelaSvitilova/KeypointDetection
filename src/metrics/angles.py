import numpy as np
from .base_metric import BaseMetric
from .utils import heatmaps_to_keypoints, scale_keypoints

class CobbAngle(BaseMetric):
    def __init__(self):
        self.error_sum = 0.0
        self.total_points = 0

    def _compute_cobb(self, p1, p2, q1, q2):
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        v2 = np.array([q2[0] - q1[0], q2[1] - q1[1]], dtype=float)
        dot = np.sum(v1 * v2, axis=1)
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0] 
        return np.degrees(np.atan2(det, dot))

    def update(self, preds, targets, orig_height, orig_width, c2_bl=0, c2_br=1, c7_bl=21, c7_br=22, **kwargs):
        _, heatmaps_height, heatmaps_width = preds[0].shape

        preds_keypoints = heatmaps_to_keypoints(preds)
        keypoints_preds = scale_keypoints(preds_keypoints, [orig_height, orig_width], [heatmaps_height, heatmaps_width])
        keypoints_targets = scale_keypoints(targets, [orig_height, orig_width], [heatmaps_height, heatmaps_width])

        cobb_preds = self._compute_cobb(
            keypoints_preds[:, c2_bl, :], 
            keypoints_preds[:, c2_br, :], 
            keypoints_preds[:, c7_bl, :], 
            keypoints_preds[:, c7_br, :]
        )
        cobb_targets = self._compute_cobb(
            keypoints_targets[:, c2_bl, :], 
            keypoints_targets[:, c2_br, :], 
            keypoints_targets[:, c7_bl, :], 
            keypoints_targets[:, c7_br, :]
        )

        error = (cobb_preds - cobb_targets + 180) % 360 - 180

        self.error_sum += np.mean(error)
        self.total_points +=1

        print(self.error_sum)
        

    def compute(self):
        return self.error_sum / self.total_points if self.total_points > 0 else 0.0

    def reset(self):
        self.error_sum = 0.0
        self.total_points = 0