from .base_metric import BaseMetric

class PCK(BaseMetric):
    """Percentage of Correct Keypoints (threshold normalized 0-1)"""
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        dist = ((preds - targets)**2).sum(dim=2).sqrt()  # Euclidean distance
        self.correct += (dist < self.threshold).sum().item()
        self.total += dist.numel()

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0
