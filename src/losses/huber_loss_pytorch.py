import torch
import torch.nn as nn
from .base_loss import BaseLoss

class HuberLossPytorch(BaseLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.HuberLoss()

    def __call__(self, preds, targets):
        return self.loss_fn(preds, targets)