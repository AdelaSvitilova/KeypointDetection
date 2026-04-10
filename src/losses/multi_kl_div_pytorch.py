import torch
from .base_loss import BaseLoss
from .kl_div_loss_pytorch import KLDivLossPytorch


class MultiKLDivLossPytorch(BaseLoss):
    """
    Applies KLDiv-based heatmap loss over multiple prediction stacks.

    Expected shapes:
        preds:   (B, S, K, H, W)
        targets: (B, K, H, W)
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = KLDivLossPytorch()

    def __call__(self, preds, targets, **kwargs):
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        return loss