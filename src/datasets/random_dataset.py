import numpy as np
from .base_dataset import BaseDataset

class RandomDataset(BaseDataset):
    """
    Generuje náhodná data pro test keypoint detection:
    - obrázky: [3, H, W]
    - keypoints: [num_keypoints, 2] (x, y) normalizované 0-1
    """
    def __init__(self, num_samples=100, num_keypoints=5, H=256, W=256):
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.H = H
        self.W = W

        self.X = np.random.randn(num_samples, 3, H, W).astype(np.float32)
        self.y = np.random.rand(num_samples, num_keypoints, 2).astype(np.float32)  # normalized coordinates

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
