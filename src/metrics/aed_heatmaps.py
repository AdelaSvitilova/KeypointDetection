from .base_metric import BaseMetric
from .aed import AED
from .utils import heatmaps_to_keypoints

class AEDHeatmaps(BaseMetric):
    def __init__(self):
        self.metric = AED()

    def update(self, preds, targets, **kwargs):
        # preds: heatmaps
        preds_kp = heatmaps_to_keypoints(preds)
        self.metric.update(preds_kp, targets)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
