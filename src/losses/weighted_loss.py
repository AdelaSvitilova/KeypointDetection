import torch
import torch.nn as nn
from .base_loss import BaseLoss

class WeightedLoss(BaseLoss):
    def __init__(self, losses, weights):
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError(
                f"losses and weights must have same length, "
                f"got {len(losses)} and {len(weights)}"
            )

        self.losses = losses
        self.weights = weights

    def __call__(self, preds, targets):
        total = 0
        for loss, w in zip(self.losses, self.weights):
            total += w * loss(preds, targets)
        return total
