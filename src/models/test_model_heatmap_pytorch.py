import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import TorchModel

class TestModelHeatmapPytorch(TorchModel):
    """
    Jednoduchý plně konvoluční model pro predikci heatmap keypointů.
    Vstup: [batch_size, input_channels, H, W]
    Výstup: [batch_size, num_keypoints, H/4, W/4] (downsampled)
    """
    def __init__(self, input_channels=1, num_keypoints=5):
        super().__init__()
        self._input_channels = input_channels
        self._num_keypoints = num_keypoints

        # Backbone: několik konvolučních bloků s downsamplingem
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Head: 1x1 konvoluce na požadovaný počet keypointů
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        # (Volitelně lze přidat upsampling, pokud chceš full-res heatmapy)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, num_keypoints, H/4, W/4]
        """
        x = self.backbone(x)
        x = self.head(x)
        # pokud chceš plnou velikost HxW, odkomentuj:
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def num_keypoints(self):
        return self._num_keypoints
