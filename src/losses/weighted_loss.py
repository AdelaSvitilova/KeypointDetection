import torch
import torch.nn as nn
from .base_loss import BaseLoss


class WeightedLoss(BaseLoss):
    """
    Weighted combination of multiple loss functions.

    Applies a weighted sum:
        total_loss = sum(w_i * loss_i(preds, targets, keypoint_targets))
    where each loss is responsible for handling its own input types and shapes.
    """

    def __init__(self, losses, weights):
        """
        Args:
            losses: List of loss callables; each must accept
                    (preds, targets=..., keypoint_targets=...).
            weights: List of weights (same length as losses).
        """
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError(
                f"losses and weights must have same length, "
                f"got {len(losses)} and {len(weights)}"
            )

        self.losses = losses
        self.weights = weights

    def __call__(self, preds, targets, keypoint_targets):
        """
        Compute weighted sum of all losses.

        Args:
            preds: Stacked predictions; e.g., (B, N, K, H, W) or (B, N, K, 2).
            targets: Optional target tensor, depending on individual loss (e.g., logits/heatmaps).
            keypoint_targets: Ground‑truth keypoints; e.g., (B, K, 2) or (B, K, H, W).

        Returns:
            Scalar total loss (weighted sum over all components).
        """
        total = 0
        for loss, w in zip(self.losses, self.weights):
            total += w * loss(
                preds, targets=targets, keypoint_targets=keypoint_targets
            )
        return total