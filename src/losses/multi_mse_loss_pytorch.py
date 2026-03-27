import torch
import torch.nn as nn
from .base_loss import BaseLoss

class MultiMSELossPytorch(BaseLoss):
    """
    PyTorch implementation of Multi-stack Mean Squared Error (MSE) loss.

    Computes MSE loss separately for each stack of predictions against the same target,
    then averages the results.

    Expected shapes:
        preds:   [B, N, C, H, W]  (predictions for N stacks)
        targets: [B, C, H, W]      (single ground truth shared for all stacks)

    This class wraps PyTorch's nn.MSELoss and provides a consistent interface
    through the BaseLoss abstract class.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute the average MSE loss over all prediction stacks.

        Parameters:
            preds (torch.Tensor): Predicted outputs with shape [B, N, C, H, W].
            targets (torch.Tensor): Single ground truth output with shape [B, C, H, W].

        Returns:
            torch.Tensor: Mean MSE loss averaged over all stacks.
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]
            loss = self.loss_fn(stack_preds, targets)
            losses.append(loss)

        #print([l.item() for l in losses])
        loss = torch.stack(losses).mean()
        return loss
