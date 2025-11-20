import torch
import torch.nn as nn
from .base_model import TorchModel

class TestModelPytorch(TorchModel):
    def __init__(self, input_channels=1, num_keypoints=5):
        super().__init__()
        self._input_channels = input_channels
        self._num_keypoints = num_keypoints

        # Jednoduchý konvoluční blok
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self._input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
        )

        # Zploštění pro lineární vrstvu
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 64 * 64, self._num_keypoints * 3)  # x, y, visibility

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        # ✅ přetvoření na [batch_size, num_keypoints, 3]
        x = x.view(x.size(0), self._num_keypoints, 3)
        return x

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def num_keypoints(self):
        return self._num_keypoints
