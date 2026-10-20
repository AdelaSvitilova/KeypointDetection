"""
CPN (Cascaded Pyramid Network) for keypoint detection.

Based on:
https://github.com/GengDavid/pytorch-cpn

Original paper:
Chen et al., Cascaded Pyramid Network for Multi-Person Pose Estimation.
https://arxiv.org/abs/1711.07319
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import TorchModel


def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + residual)


class ResidualStage(nn.Module):
    def __init__(self, in_ch, out_ch, blocks, stride):
        super().__init__()

        layers = [ResidualBlock(in_ch, out_ch, stride)]

        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GlobalNet(nn.Module):
    """Top-down pyramid feature extractor."""

    def __init__(self, channels=(64, 128, 256, 512), out_channels=256):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in channels
        ])

        self.smooth = nn.ModuleList([
            nn.Sequential(
                _conv3x3(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in channels
        ])

    def forward(self, features):
        pyramid = [None] * len(features)

        top = self.lateral[-1](features[-1])
        pyramid[-1] = self.smooth[-1](top)

        for i in range(len(features) - 2, -1, -1):
            top = self.lateral[i](features[i]) + F.interpolate(
                top,
                size=features[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[i] = self.smooth[i](top)

        return pyramid


class RefineNet(nn.Module):
    """Combine multiscale features and predict keypoint heatmaps."""

    def __init__(self, channels=256, num_keypoints=23):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
        )

        self.output = nn.Conv2d(
            channels,
            num_keypoints,
            kernel_size=1,
        )

    def forward(self, pyramid):
        size = pyramid[0].shape[-2:]

        features = [pyramid[0]]

        for feature in pyramid[1:]:
            features.append(
                F.interpolate(
                    feature,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        x = torch.cat(features, dim=1)
        x = self.reduce(x)

        return self.output(x)


class CPNPytorch(TorchModel):
    """CPN-style model compatible with the repository's heatmap pipeline.

    If use_intermediate_loss=True, returns:
        (B, 2, K, H, W)
        Index 0: intermediate prediction from GlobalNet.
        Index 1: final prediction from RefineNet.

    Otherwise returns:
        (B, K, H, W)
    """

    def __init__(
        self,
        width,
        pyramid_channels,
        stage_blocks,
        input_channels,
        num_keypoints,
        use_intermediate_loss=False,
        **kwargs,
    ):
        super(CPNPytorch, self).__init__()

        self.use_intermediate_loss = use_intermediate_loss

        # Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                width,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = ResidualStage(
            width,
            width * 2,
            stage_blocks[0],
            stride=1,
        )
        self.stage2 = ResidualStage(
            width * 2,
            width * 4,
            stage_blocks[1],
            stride=2,
        )
        self.stage3 = ResidualStage(
            width * 4,
            width * 8,
            stage_blocks[2],
            stride=2,
        )
        self.stage4 = ResidualStage(
            width * 8,
            width * 16,
            stage_blocks[3],
            stride=2,
        )

        channels = (
            width * 2,
            width * 4,
            width * 8,
            width * 16,
        )

        # Cascaded Pyramid Network
        self.global_net = GlobalNet(channels, pyramid_channels)

        self.refine_net = RefineNet(pyramid_channels, num_keypoints)

        # Optional intermediate prediction from GlobalNet.
        if self.use_intermediate_loss:
            self.intermediate_output = nn.Conv2d(
                pyramid_channels,
                num_keypoints,
                kernel_size=1,
            )

    def forward(self, x):
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        pyramid = self.global_net([c2, c3, c4, c5])

        # Final prediction from RefineNet.
        final_heatmaps = self.refine_net(pyramid)

        if self.use_intermediate_loss:
            # Intermediate prediction from the highest-resolution
            # feature map produced by GlobalNet.
            intermediate_heatmaps = self.intermediate_output(pyramid[0])

            # Shape: (B, 2, K, H, W)
            # Order: intermediate prediction, final prediction.
            return torch.stack(
                [intermediate_heatmaps, final_heatmaps],
                dim=1,
            )

        # Original behavior: (B, K, H, W)
        return final_heatmaps