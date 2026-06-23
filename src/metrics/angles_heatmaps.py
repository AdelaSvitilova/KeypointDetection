import numpy as np
from .base_metric import BaseMetric
from .angles import CobbAngle
from .utils import heatmaps_to_keypoints

class CobbAngleHeatmaps(BaseMetric):
    def __init__(self):
        self.metric = CobbAngle()

    def update(self, preds, targets, orig_height, orig_width, predict_height=None, predict_width=None, c2_bl=0, c2_br=1, c7_bl=21, c7_br=22, **kwargs):
        preds_keypoints = heatmaps_to_keypoints(preds)

        _, heatmaps_height, heatmaps_width = preds[0].shape
        
        self.metric.update(preds_keypoints, targets, orig_height, orig_width, predict_height=heatmaps_height, predict_width=heatmaps_width, c2_bl=c2_bl, c2_br=c2_br, c7_bl=c7_bl, c7_br=c7_br)
        

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()