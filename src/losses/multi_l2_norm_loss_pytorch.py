import torch
import torch.nn as nn
from .base_loss import BaseLoss
from .l2_norm_loss_pytorch import L2NormLossPytorch


class MultiL2NormLossPytorch(BaseLoss):
    """
    Multi‑stack L2‑norm loss for keypoint regression.

    Expects predictions stacked along dim=1:
        (B, N, K, H, W) or (B, N, K, 2)
    and applies L2‑norm loss per stack (after conversion to keypoints),
    then averages the loss over stacks.
    """

    def __init__(self):
        """
        Initialize the multi‑stack L2‑norm loss using L2NormLossPytorch.
        """
        super().__init__()
        self.loss_fn = L2NormLossPytorch()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute mean L2‑norm loss over all stacks.

        Args:
            preds: Stacked predictions of shape (B, N, K, H, W) or (B, N, K, 2),
                   where N is the number of stacks and K is the number of keypoints.
            targets: Ground‑truth keypoint targets of shape (B, K, 2) or (B, K, H, W).

        Returns:
            Scalar loss (mean over stacks).
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]  # (B, K, H, W) or (B, K, 2)
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        # Stack and average over stacks.
        loss = torch.stack(losses).mean()
        return loss