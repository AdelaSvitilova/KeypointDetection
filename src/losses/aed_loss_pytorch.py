import torch
import torch.nn as nn
from .base_loss import BaseLoss

class AEDLossPytorch(BaseLoss):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def __call__(self, preds, targets):
        diff = preds - targets
        
        dist = torch.norm(diff, p=2, dim=1)
        
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist
