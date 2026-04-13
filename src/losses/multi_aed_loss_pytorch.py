import torch
import torch.nn as nn
from .base_loss import BaseLoss
from .aed_loss_pytorch import AEDLossPytorch


class MultiAEDLossPytorch(BaseLoss):
    """
    Multi‑stack AED loss for stacked‑hourglass‑style outputs.

    Expects predictions stacked along dim=1 and applies AED loss per stack,
    then averages over stacks.
    """

    def __init__(self):
        """
        Initialize the multi‑stack AED loss using AEDLossPytorch.
        """
        super().__init__()
        self.loss_fn = AEDLossPytorch()

    def __call__(self, preds, keypoint_targets, **kwargs):
        """
        Compute mean AED loss over all stacks.

        Args:
            preds: Stacked predictions of shape (B, N, K, H, W) or (B, N, K, 2),
                   where N is the number of stacks.
            keypoint_targets: Ground‑truth keypoints (B, K, 2) or (B, K, H, W).

        Returns:
            Scalar loss (mean over stacks).
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]  # (B, K, H, W) or (B, K, 2)
            loss = self.loss_fn(stack_preds, keypoint_targets, **kwargs)
            losses.append(loss)

        # Stack and average over stacks.
        loss = torch.stack(losses).mean()
        return loss