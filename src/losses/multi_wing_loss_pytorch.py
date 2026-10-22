
import torch
from .base_loss import BaseLoss
from .wing_loss_pytorch import WingLossPytorch


class MultiWingLossPytorch(BaseLoss):
    """Multi-stack Wing loss for stacked-hourglass-style outputs.

    Expects predictions stacked along dim=1 and applies Wing loss
    to each stack, then averages the losses over all stacks.
    """

    def __init__(self, width=10.0, epsilon=2.0):
        """Initialize the multi-stack Wing loss.

        Args:
            width: Threshold between logarithmic and linear loss.
            epsilon: Controls the curvature of the logarithmic part.
        """
        super().__init__()
        self.loss_fn = WingLossPytorch(
            width=width,
            epsilon=epsilon,
        )

    def __call__(self, preds, targets, **kwargs):
        """Compute mean Wing loss over all stacks.

        Args:
            preds: Stacked predictions with shape (B, N, ...),
                where N is the number of stacks.
            targets: Target tensor matching the shape of preds[:, i].

        Returns:
            Scalar loss averaged over all stacks.
        """
        num_stacks = preds.size(1)

        losses = []
        for i in range(num_stacks):
            stack_preds = preds[:, i]
            loss = self.loss_fn(stack_preds, targets, **kwargs)
            losses.append(loss)

        # Stack and average the losses over all stacks.
        return torch.stack(losses).mean()
