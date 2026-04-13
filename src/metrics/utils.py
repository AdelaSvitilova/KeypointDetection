import numpy as np


def heatmaps_to_keypoints(heatmaps):
    """
    Convert heatmaps to discrete keypoints with visibility flags.

    Args:
        heatmaps: Array of shape (B, K, H, W).

    Returns:
        Keypoints of shape (B, K, 3), where the last dimension is (x, y, visible).
    """
    B, K, H, W = heatmaps.shape

    # 1. Flatten spatial dimensions.
    heatmaps_flat = heatmaps.reshape(B, K, -1)  # (B, K, H*W)

    # 2. Get index of maximum per keypoint.
    idx = np.argmax(heatmaps_flat, axis=2)  # (B, K)

    # 3. Convert flat index to (x, y).
    y = idx // W
    x = idx % W

    # 4. Visibility flag (assume all visible).
    visible = np.ones((B, K), dtype=heatmaps.dtype)

    # 5. Stack (x, y, visible) along last dim.
    keypoints = np.stack([x, y, visible], axis=2)  # (B, K, 3)

    return keypoints