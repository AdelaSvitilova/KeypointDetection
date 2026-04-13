import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class AEDLossPytorch(BaseLoss):
    """Average Euclidean Distance (AED) loss handling both heatmaps and keypoint vectors.

    Given predictions and targets, this loss:
        - converts heatmaps to soft‑argmax keypoints if needed,
        - computes Euclidean distance per keypoint,
        - reduces the result over the batch.
    """

    def __init__(self, reduction="mean", beta=100.0):
        """
        Args:
            reduction: Reduction over batch; one of "mean", "sum", or None.
            beta: Softmax temperature; higher values make the distribution sharper.
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def _is_heatmap(self, x):
        """Returns True if x is a heatmap tensor (B, K, H, W)."""
        return x.dim() == 4

    def _soft_argmax(self, heatmaps):
        """
        Convert heatmaps (B, K, H, W) to soft keypoints (B, K, 2).

        Args:
            heatmaps: Tensor of shape (B, K, H, W).

        Returns:
            Soft keypoints tensor of shape (B, K, 2).
        """
        B, K, H, W = heatmaps.shape

        # Flatten spatial dimension.
        heatmaps = heatmaps.view(B, K, -1)  # (B, K, H*W)
        probs = F.softmax(heatmaps * self.beta, dim=2)

        # Create coordinate grid (H, W).
        y_coords = torch.linspace(0, H - 1, H, device=heatmaps.device)
        x_coords = torch.linspace(0, W - 1, W, device=heatmaps.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        xx = xx.reshape(-1)  # (H*W)
        yy = yy.reshape(-1)

        # Compute expected x, y per keypoint.
        exp_x = torch.sum(probs * xx, dim=2)
        exp_y = torch.sum(probs * yy, dim=2)

        keypoints = torch.stack([exp_x, exp_y], dim=2)  # (B, K, 2)
        return keypoints

    def _to_keypoints(self, x):
        """Convert input to keypoint coordinates, using _soft_argmax for heatmaps."""
        if self._is_heatmap(x):
            return self._soft_argmax(x)
        return x  # Already (B, K, 2)

    def __call__(self, preds, keypoint_targets, **kwargs):
        """
        Compute AED loss between predictions and targets.

        Args:
            preds: Predictions; either heatmaps (B, K, H, W) or keypoints (B, K, 2).
            keypoint_targets: Ground‑truth keypoints (B, K, 2).

        Returns:
            Scalar loss (if reduction is "mean" or "sum") or tensor (B, K).
        """
        preds_kp = self._to_keypoints(preds)
        targets_kp = self._to_keypoints(keypoint_targets)

        if preds_kp.shape != targets_kp.shape:
            raise ValueError(
                f"Shape mismatch: {preds_kp.shape} vs {targets_kp.shape}"
            )

        diff = preds_kp - targets_kp  # (B, K, 2)
        dist = torch.norm(diff, p=2, dim=2)  # (B, K)

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        else:
            return dist