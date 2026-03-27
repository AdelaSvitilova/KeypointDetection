import torch
import torch.nn as nn
from .base_loss import BaseLoss

class MSELossPytorch(BaseLoss):
    """
    PyTorch implementation of Mean Squared Error (MSE) loss for keypoints.

    Predictions and targets should be torch.Tensor with shape:
    [batch_size, num_keypoints, 2] or [batch_size, num_keypoints, 3].
    The predictions tensor must have requires_grad=True for backpropagation.

    This class wraps PyTorch's nn.MSELoss and provides a consistent interface
    through the BaseLoss abstract class.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute the MSE loss between predictions and target keypoints.

        Parameters:
            preds (torch.Tensor): Predicted keypoints.
            targets (torch.Tensor): Ground truth keypoints.

        Returns:
            torch.Tensor: Computed MSE loss.
        """
        return self.loss_fn(preds, targets)
