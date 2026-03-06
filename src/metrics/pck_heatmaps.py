from .base_metric import BaseMetric
from .pck import PCK
from .utils import heatmaps_to_keypoints

class PCKHeatmaps(BaseMetric):
    def __init__(self, threshold=0.5):
        self.metric = PCK(threshold)

    def update(self, preds, targets, norm_coefficient=25):
        # preds: heatmaps
        preds_kp = heatmaps_to_keypoints(preds)
        self.metric.update(preds_kp, targets, norm_coefficient)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
