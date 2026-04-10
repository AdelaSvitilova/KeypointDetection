import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss

class KLDivLossPytorch(BaseLoss):
    """
    KL divergence loss for keypoint heatmaps.

    Expected shapes:
        preds:   (B, K, H, W)  - raw logits from model
        targets: (B, K, H, W)  - heatmaps (not necessarily normalized)
    """

    def __init__(self, reduction="batchmean", eps=1e-8):
        self.kl = nn.KLDivLoss(reduction=reduction)
        self.eps = eps

    def __call__(self, preds, targets, **kwargs):
        B, K, H, W = preds.shape

        # Flatten spatial dimensions
        preds = preds.view(B, K, -1)      # (B, K, H*W)
        targets = targets.view(B, K, -1)  # (B, K, H*W)

        # ---- preds → log probabilities ----
        log_probs = F.log_softmax(preds, dim=-1)

        # ---- targets → normalized probabilities ----
        targets = targets + self.eps  # avoid division by zero
        targets = targets / targets.sum(dim=-1, keepdim=True)

        # ---- compute KL divergence ----
        loss = self.kl(log_probs, targets)

        return loss