import torch
import torch.nn as nn
import dsntnn
from .base_model import TorchModel

class DSNTHead(TorchModel):
    """Calculates coordinate expectations from heatmaps using DSNT.

    This module applies a spatial softmax to the input heatmaps and uses
    the Differentiable Spatial to Numerical Transform (DSNT) to obtain
    the expected 2D coordinates in pixel space.
    """

    def forward(self, heatmaps):
        """Calculate coordinates from one or multiple heatmap predictions.

        Args:
            heatmaps: Tensor of shape (B, K, H, W) or
                (B, N, K, H, W), where N is the number of predictions.

        Returns:
            Coordinates of shape (B, K, 2) for one prediction,
            or (B, N, K, 2) for multiple predictions.
        """
        # Process all predictions with the same DSNT transformation.
        if heatmaps.ndim == 5:
            batch_size, num_predictions, num_keypoints, height, width = (
                heatmaps.shape
            )

            heatmaps = heatmaps.reshape(
                batch_size * num_predictions,
                num_keypoints,
                height,
                width,
            )

            multiple_predictions = True

        elif heatmaps.ndim == 4:
            multiple_predictions = False

        else:
            raise ValueError(
                "Expected heatmaps with shape (B, K, H, W) "
                "or (B, N, K, H, W)."
            )

        # Normalize each heatmap into a spatial probability distribution.
        heatmaps = dsntnn.flat_softmax(heatmaps)

        # Calculate the expected coordinates in the normalized range [-1, 1].
        coords = dsntnn.dsnt(heatmaps)

        # Map normalized coordinates to heatmap pixel coordinates.
        height, width = heatmaps.shape[-2:]
        coords_x = (coords[..., 0] + 1) * (width - 1) / 2
        coords_y = (coords[..., 1] + 1) * (height - 1) / 2

        coords = torch.stack((coords_x, coords_y), dim=-1)

        if multiple_predictions:
            coords = coords.reshape(
                batch_size,
                num_predictions,
                num_keypoints,
                2,
            )

        return coords


class DSNTModelPytorch(TorchModel):
    """A composite PyTorch model combining a base network with a DSNT head.

    Args:
        base_model (torch.nn.Module): The backbone neural network that extracts
            features and predicts heatmaps.
    """

    def __init__(self, base_model):
        """Initializes the DSNTModelPytorch with the given base model."""
        super().__init__()
        self.base_model = base_model
        self.dsnt_head = DSNTHead()

    def forward(self, x):
        """Forward pass of the full model.

        Args:
            x (torch.Tensor): Input tensor (usually batched images) of shape 
                `(B, C_in, H_in, W_in)`.

        Returns:
            torch.Tensor: Predicted 2D keypoint coordinates of shape `(B, K, 2)`,
                where K is the number of keypoints/channels.
        """
        # Generate heatmaps using the base model
        heatmaps = self.base_model(x)
        
        # Predict coordinates from the heatmaps using the DSNT head
        coords = self.dsnt_head(heatmaps)

        return coords