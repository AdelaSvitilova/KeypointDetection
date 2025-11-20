import torch
import torch.nn as nn
from .base_loss import BaseLoss

class KeypointMSELossPytorch(BaseLoss):
    """
    PyTorch-based MSE loss pro keypointy.
    Preds a targets by měly být torch.Tensor: [batch_size, num_keypoints, 2] (nebo [batch_size, num_keypoints, 3])
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, preds, targets):
        # preds a targets musí být torch.Tensor s requires_grad (preds)
        return self.loss_fn(preds, targets)
