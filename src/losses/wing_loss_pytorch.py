
import torch
from .base_loss import BaseLoss


class WingLossPytorch(BaseLoss):
    """Wing loss for keypoint coordinate regression.

    Applies a logarithmic penalty to small errors and a linear
    penalty to large errors, as proposed in:
    'Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks'.
    """

    def __init__(self, width=10.0, epsilon=2.0):
        """Initialize the Wing loss parameters.

        Args:
            width: Threshold between logarithmic and linear loss.
            epsilon: Controls the curvature of the logarithmic part.
        """
        super().__init__()

        if width <= 0:
            raise ValueError("width must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.width = width
        self.epsilon = epsilon

        priint(width, epsilon)

        # Ensures continuity between the logarithmic and linear parts.
        self.constant = width * (
            1.0 - torch.log1p(torch.tensor(width / epsilon)).item()
        )

    def __call__(self, preds, targets, **kwargs):
        """Compute Wing loss between predictions and targets.

        Args:
            preds: Predictions tensor of shape (B, ...).
            targets: Target tensor with the same shape as preds.

        Returns:
            Scalar Wing loss averaged over all elements.
        """
        if preds.shape != targets.shape:
            raise ValueError(
                "preds and targets must have the same shape."
            )

        # Absolute error between predicted and target coordinates.
        error = torch.abs(preds - targets)

        # Logarithmic penalty for small errors.
        logarithmic_loss = self.width * torch.log1p(
            error / self.epsilon
        )

        # Linear penalty for large errors.
        linear_loss = error - self.constant

        # Apply the appropriate loss according to the error magnitude.
        loss = torch.where(
            error < self.width,
            logarithmic_loss,
            linear_loss,
        )

        return loss.mean()
