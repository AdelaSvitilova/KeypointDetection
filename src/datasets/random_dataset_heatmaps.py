import numpy as np
from .base_dataset import BaseDataset

def gaussian_2d(H, W, center, sigma=3):
    """
    Vytvoří 2D Gaussovu heatmapu.
    center: (x, y) v pixelech
    sigma: rozptyl Gaussovy funkce
    """
    x = np.arange(0, W, 1, float)
    y = np.arange(0, H, 1, float)[:, np.newaxis]
    cx, cy = center
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

class RandomHeatmapDataset(BaseDataset):
    """
    Generuje náhodná data pro test keypoint detection:
    - obrázky: [3, H, W]
    - heatmapy: [num_keypoints, H, W]
    """
    def __init__(self, num_samples=20, num_keypoints=5, H=256, W=256, sigma=3, split=None):
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.H = H
        self.W = W
        self.sigma = sigma

        self.X = np.random.randn(num_samples, 3, H, W).astype(np.float32)
        # generujeme náhodné keypointy v pixelech
        self.keypoints = np.random.rand(num_samples, num_keypoints, 2)
        self.keypoints[:, :, 0] *= W  # x
        self.keypoints[:, :, 1] *= H  # y

        # vytváříme heatmapy
        self.y = np.zeros((num_samples, num_keypoints, H, W), dtype=np.float32)
        for i in range(num_samples):
            for j in range(num_keypoints):
                self.y[i, j] = gaussian_2d(H, W, self.keypoints[i, j], sigma=self.sigma)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
