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
        """Forward pass to calculate coordinates from heatmaps.

        Args:
            heatmaps (torch.Tensor): Input heatmaps. Can be 4D `(B, C, H, W)`
                or 5D `(B, N, C, H, W)`. If 5D, only the last heatmap in
                the sequence is used.

        Returns:
            torch.Tensor: The predicted 2D coordinates in pixel space
                of shape `(B, C, 2)`.
        """
        # Select the last heatmap if the input is a sequence of heatmaps (5D tensor)
        if heatmaps.ndim == 5:
            heatmaps = heatmaps[:, -1, :, :, :].contiguous()

        # Normalize the heatmaps so that each channel sums to 1 (spatial probability distribution)
        heatmaps = dsntnn.flat_softmax(heatmaps)

        # Calculate the expected coordinates in the normalized range [-1, 1]
        coords = dsntnn.dsnt(heatmaps)

        # Extract the height and width of the heatmaps
        h, w = heatmaps.shape[-2:]

        # Map the coordinates from [-1, 1] back to pixel space [0, W-1] and [0, H-1]
        coords_x = (coords[..., 0] + 1) * (w - 1) / 2
        coords_y = (coords[..., 1] + 1) * (h - 1) / 2

        # Stack x and y coordinates along the last dimension
        coords = torch.stack((coords_x, coords_y), dim=-1)

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