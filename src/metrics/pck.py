import numpy as np
from .base_metric import BaseMetric

class PCK(BaseMetric):
    """Percentage of Correct Keypoints (threshold normalized 0-1)"""
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        preds_xy = preds[..., :2]
        targets_xy = targets[..., :2]

        # Euclidean distance
        dist = np.sqrt(np.sum((preds_xy - targets_xy) ** 2, axis=2))  # (B, K)

        self.correct += np.sum(dist < self.threshold)
        self.total += dist.size

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0
