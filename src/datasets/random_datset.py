import numpy as np
from .base_dataset import BaseDataset

class RandomDataset(BaseDataset):
    def __init__(self, num_samples=20, transform=None, heatmaps=None, 
                 num_keypoints=5, input_size=(256,256), 
                 output_size=(64,64), sigma=3):
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.transform = transform
        self.to_heatmaps = heatmaps

        H_out, W_out = self.output_size
        # generujeme keypointy přímo ve výstupním prostoru jako integer
        self.keypoints = np.random.rand(num_samples, num_keypoints, 2)
        self.keypoints[:, :, 0] = (self.keypoints[:, :, 0] * W_out).astype(int)  # x
        self.keypoints[:, :, 1] = (self.keypoints[:, :, 1] * H_out).astype(int)  # y

        v = np.ones((num_samples, num_keypoints, 1), dtype=np.float32)
        self.keypoints = np.concatenate([self.keypoints, v], axis=2)

        # obrázky v input_size prostoru
        H_in, W_in = self.input_size
        self.images = np.random.randn(num_samples, 3, H_in, W_in).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        keypoints = self.keypoints[idx].copy()

        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        heatmaps = None

        if self.to_heatmaps:
            heatmaps = self.to_heatmaps(keypoints, self.output_size)

        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps
        }
