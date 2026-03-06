import numpy as np
from .base_dataset import BaseDataset


class RandomDataset(BaseDataset):
    """Dataset that generates random images and keypoints for testing purposes.

    This dataset is intended for debugging and validating data pipelines
    for keypoint detection or pose estimation models without relying on
    real annotated data.

    Random RGB images are generated in the input resolution, while keypoints
    are generated directly in the output (heatmap) resolution. Each keypoint
    consists of (x, y, visibility).

    Optionally, transforms can be applied to both images and keypoints,
    and keypoints can be converted to heatmaps using a provided callable.
    """

    def __init__(
        self,
        num_samples: int = 20,
        transform=None,
        heatmaps=None,
        num_keypoints: int = 5,
        input_size: tuple = (256, 256),
        output_size: tuple = (64, 64),
        sigma: int = 3,
    ):
        """Initialize the RandomDataset.

        Args:
            num_samples (int): Number of samples in the dataset.
            transform (callable, optional): Optional transform applied to
                both image and keypoints.
            heatmaps (callable, optional): Function that converts keypoints
                into heatmaps.
            num_keypoints (int): Number of keypoints per sample.
            input_size (tuple): Spatial resolution of the input images
                as (height, width).
            output_size (tuple): Spatial resolution of the output heatmaps
                as (height, width).
            sigma (int): Standard deviation used for heatmap generation.
        """
        # Store initialization parameters
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.transform = transform
        self.to_heatmaps = heatmaps

        H_out, W_out = self.output_size

        # Generate random keypoints in output space (integers)
        self.keypoints = np.random.rand(num_samples, num_keypoints, 2)
        self.keypoints[:, :, 0] = (self.keypoints[:, :, 0] * W_out).astype(int)  # x coordinates
        self.keypoints[:, :, 1] = (self.keypoints[:, :, 1] * H_out).astype(int)  # y coordinates

        # Add visibility channel (1 = visible)
        visibility = np.ones((num_samples, num_keypoints, 1), dtype=np.float32)
        self.keypoints = np.concatenate([self.keypoints, visibility], axis=2)

        H_in, W_in = self.input_size
        # Generate random RGB images in input space
        self.images = np.random.randn(num_samples, 3, H_in, W_in).astype(np.float32)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - image (np.ndarray): The input image tensor.
                - keypoints (np.ndarray): Array of keypoints with shape
                  (num_keypoints, 3).
                - heatmaps (np.ndarray or None): Generated heatmaps if
                  a heatmap function is provided, otherwise None.
        """
        # Copy image and keypoints to avoid modifying originals
        image = self.images[idx].copy()
        keypoints = self.keypoints[idx].copy()

        # Apply optional transforms
        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        # Optionally convert keypoints to heatmaps
        heatmaps = None
        if self.to_heatmaps:
            heatmaps = self.to_heatmaps(keypoints, self.output_size)

        # Return dictionary of data
        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "norm_coefficient": 25
        }
