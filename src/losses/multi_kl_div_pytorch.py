import torch
from .base_loss import BaseLoss
from .kl_div_loss_pytorch import KLDivLossPytorch


class MultiKLDivLossPytorch(BaseLoss):
    """
    Multi‑stack KL‑divergence loss for keypoint heatmaps.

    Applies KLDivLoss over multiple prediction stacks (dim=1) and averages the loss.
    Shape assumptions:
        preds:   (B, S, K, H, W)  # logits from stacked predictions
        targets: (B, K, H, W)     # ground‑truth heatmap probabilities
    """

    def __init__(self):
        """
        Initialize the multi‑stack KL divergence loss using KLDivLossPytorch.
        """
        super().__init__()
        self.loss_fn = KLDivLossPytorch()

    def __call__(self, preds, targets, **kwargs):
        """
        Compute mean KL divergence loss over all stacks.

        Args:
            preds: Stacked logits of shape (B, S, K, H, W),
                   where S is the number of stacks and K is the number of keypoints.
            targets: Target heatmaps of shape (B, K, H, W).

        Returns:
            Scalar loss (mean over stacks).
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]  # (B, K, H, W)
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        # Stack and average over stacks.
        loss = torch.stack(losses).mean()
        return loss