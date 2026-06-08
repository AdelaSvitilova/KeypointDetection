import numpy as np


def keypoints_to_heatmaps_np(keypoints, size=(256, 256), sigma=2):
    """
    Converts keypoints into Gaussian heatmaps.

    Each keypoint is represented as a 2D Gaussian centered at its (x, y)
    coordinate. The output is a stack of heatmaps, one per keypoint.

    Args:
        keypoints (np.ndarray): Array of shape [K, 2] or [K, 3], where each row
            is (x, y) or (x, y, visibility/confidence).
        size (tuple): Output heatmap size as (height, width).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Heatmaps of shape [K, H, W], one heatmap per keypoint.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    K = keypoints.shape[0]

    height, width = size  # unpack output spatial dimensions

    # Create coordinate grid for vectorized Gaussian computation
    # y: [H, 1], x: [1, W]
    y = np.arange(height).reshape(-1, 1)
    x = np.arange(width).reshape(1, -1)

    heatmaps = np.zeros((K, height, width), dtype=np.float32)

    # If keypoints include visibility/confidence, drop it
    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]

    for i, (kx, ky) in enumerate(keypoints):

        # Skip invalid or out-of-bounds keypoints
        if kx < 0 or ky < 0 or kx >= width or ky >= height:
            continue

        # Compute 2D Gaussian centered at (kx, ky)
        # Broadcasting produces full HxW heatmap
        heatmap = np.exp(-((x - kx) ** 2 + (y - ky) ** 2) / (2 * sigma ** 2))

        heatmaps[i] = heatmap

    return heatmaps