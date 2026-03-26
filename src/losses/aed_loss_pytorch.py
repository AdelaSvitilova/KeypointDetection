import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss

class AEDLossPytorch(BaseLoss):

    def __init__(self, reduction='mean', beta=100.0):
        """
        beta = teplota softmaxu (vyšší = ostřejší, blíž argmaxu)
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def _is_heatmap(self, x):
        return x.dim() == 4  # (B, K, H, W)

    def _soft_argmax(self, heatmaps):
        """
        heatmaps: (B, K, H, W)
        return: (B, K, 2)
        """
        B, K, H, W = heatmaps.shape

        heatmaps = heatmaps.view(B, K, -1)  # (B, K, H*W)
        probs = F.softmax(heatmaps * self.beta, dim=2)

        # vytvoření souřadnicové mřížky
        y_coords = torch.linspace(0, H - 1, H, device=heatmaps.device)
        x_coords = torch.linspace(0, W - 1, W, device=heatmaps.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        xx = xx.reshape(-1)  # (H*W)
        yy = yy.reshape(-1)

        exp_x = torch.sum(probs * xx, dim=2)
        exp_y = torch.sum(probs * yy, dim=2)

        keypoints = torch.stack([exp_x, exp_y], dim=2)  # (B, K, 2)
        return keypoints

    def _to_keypoints(self, x):
        if self._is_heatmap(x):
            return self._soft_argmax(x)
        return x  # (B, K, 2)

    def __call__(self, preds, targets):
        preds_kp = self._to_keypoints(preds)
        targets_kp = self._to_keypoints(targets)

        if preds_kp.shape != targets_kp.shape:
            raise ValueError(f"Shape mismatch: {preds_kp.shape} vs {targets_kp.shape}")

        diff = preds_kp - targets_kp  # (B, K, 2)
        dist = torch.norm(diff, p=2, dim=2)  # (B, K)

        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist