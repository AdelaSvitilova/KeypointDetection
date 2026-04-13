import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class KLDivLossPytorch(BaseLoss):
    """
    KL divergence loss for keypoint heatmaps.

    Expected shapes:
        preds:   (B, K, H, W)  # raw logits from the model
        targets: (B, K, H, W)  # heatmaps (not necessarily normalized)
    """

    def __init__(self, reduction="batchmean", eps=1e-8):
        """
        Args:
            reduction: Reduction mode for nn.KLDivLoss; one of "batchmean", "sum", "mean", "none".
            eps: Small value added for numerical stability when normalizing targets.
        """
        super().__init__()
        self.kl = nn.KLDivLoss(reduction=reduction)
        self.eps = eps

    def __call__(self, preds, targets, **kwargs):
        """
        Compute KL divergence between predicted logits and target heatmaps.

        Args:
            preds:   Raw logits of shape (B, K, H, W).
            targets: Heatmaps of shape (B, K, H, W), interpreted as unnormalized probabilities.

        Returns:
            Scalar loss (depending on reduction setting).
        """
        B, K, H, W = preds.shape

        # Flatten spatial dimensions.
        preds = preds.view(B, K, -1)      # (B, K, H*W)
        targets = targets.view(B, K, -1)  # (B, K, H*W)

        # Convert logits to log probabilities.
        log_probs = F.log_softmax(preds, dim=-1)

        # Normalize targets into proper probabilities.
        targets = targets + self.eps  # Avoid division by zero.
        targets = targets / targets.sum(dim=-1, keepdim=True)

        # Compute KL divergence.
        loss = self.kl(log_probs, targets)

        return loss