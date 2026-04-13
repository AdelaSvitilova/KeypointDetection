import torch
import torch.nn as nn
from .base_loss import BaseLoss


class L2NormLossPytorch(BaseLoss):
    """
    L2 (Euclidean) norm loss between predictions and targets.

    Computes the L2 distance per sample (along dim=1) and optionally reduces over the batch.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction: Reduction over batch; one of "mean", "sum", or None.
        """
        super().__init__()
        self.reduction = reduction

    def __call__(self, preds, targets, **kwargs):
        """
        Compute L2 norm loss.

        Args:
            preds:   Predictions tensor of shape (B, D).
            targets: Target tensor of the same shape.

        Returns:
            Scalar loss (if mean/sum) or tensor (B,) if no reduction.
        """
        diff = preds - targets  # (B, D)

        dist = torch.norm(diff, p=2, dim=1)  # L2 norm per sample (B,)

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        else:
            return dist