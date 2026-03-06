import numpy as np
from .base_metric import BaseMetric

class AED(BaseMetric): #avarage euclidean distance
    def __init__(self):
        self.sum_dist = 0.0
        self.total_points = 0

    def update(self, preds, targets, **kwargs):
        preds_xy = preds[..., :2]
        targets_xy = targets[..., :2]

        # Euclidean distance
        dist = np.sqrt(np.sum((preds_xy - targets_xy) ** 2, axis=2))  # (B, K)

        self.sum_dist += np.sum(dist)
        self.total_points += dist.size

    def compute(self):
        return self.sum_dist / self.total_points if self.total_points > 0 else 0.0

    def reset(self):
        self.sum_dist = 0.0
        self.total_points = 0