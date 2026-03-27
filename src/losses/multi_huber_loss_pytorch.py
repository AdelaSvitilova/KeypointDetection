import torch
import torch.nn as nn
from .base_loss import BaseLoss
from .huber_loss_pytorch import HuberLossPytorch

class MultiHuberLossPytorch(BaseLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = HuberLossPytorch()

    def __call__(self, preds, targets, **kwargs):
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        #print([l.item() for l in losses])
        loss = torch.stack(losses).mean()
        return loss
