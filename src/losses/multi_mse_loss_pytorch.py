import torch
import torch.nn as nn
from .base_loss import BaseLoss

class MultiMSELossPytorch(BaseLoss):
    """
    Vrací MSE pro každou větev zvlášť.
    preds:   [B, C, H, W]
    targets: [B, N, C, H, W]  (N = počet větví = num_branches)
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, preds, targets):
        num_stack = preds.size(1)

        losses = []
        for i in range(num_stack):
            stack_preds = preds[:, i]
            loss = self.loss_fn(stack_preds, targets)
            losses.append(loss)

        #print([l.item() for l in losses])
        loss = torch.stack(losses).mean()
        return loss
