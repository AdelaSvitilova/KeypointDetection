import torch
import torch.nn as nn
from .base_loss import BaseLoss
from .huber_loss_pytorch import HuberLossPytorch


class MultiHuberLossPytorch(BaseLoss):
    """
    Multi‑stack Huber loss for stacked‑hourglass‑style outputs.

    Expects predictions stacked along dim=1 and applies Huber loss per stack,
    then averages over stacks.
    """

    def __init__(self):
        """
        Initialize the multi‑stack Huber loss using HuberLossPytorch.
        """
        super().__init__()
        self.loss_fn = HuberLossPytorch()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute mean Huber loss over all stacks.

        Args:
            preds: Stacked predictions of shape (B, N, D), where N is the number of stacks.
            targets: Target tensor of shape (B, D).

        Returns:
            Scalar loss (mean over stacks).
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]  # (B, D)
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        # Stack and average over stacks.
        loss = torch.stack(losses).mean()
        return loss