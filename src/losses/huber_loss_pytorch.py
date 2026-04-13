import torch
import torch.nn as nn
from .base_loss import BaseLoss


class HuberLossPytorch(BaseLoss):
    """Wraps PyTorch's HuberLoss for keypoint regression.

    Uses Huber (smooth L1) loss between predictions and targets,
    which is less sensitive to outliers than MSE.
    """

    def __init__(self):
        """Initialize the Huber loss module with default reduction and delta."""
        super().__init__()
        self.loss_fn = nn.HuberLoss()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute Huber loss between predictions and targets.

        Args:
            preds: Predictions tensor of shape (B, ...).
            targets: Target tensor of the same shape as preds.

        Returns:
            Scalar loss (depending on the reduction set in nn.HuberLoss).
        """
        return self.loss_fn(preds, targets)