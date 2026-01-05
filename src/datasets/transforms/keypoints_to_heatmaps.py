import numpy as np

def keypoints_to_heatmaps_np(keypoints, size=(256, 256), sigma=2):
    """
    keypoints: array [K, 2] -> (x, y) coordinates in the image
    size: tuple (height, width) of output heatmap
    sigma: Gaussian sigma

    returns: array [K, H, W]
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    K = keypoints.shape[0]

    height, width = size  # rozbal tuple

    # coordinate grid
    y = np.arange(height).reshape(-1, 1)  # shape [H, 1]
    x = np.arange(width).reshape(1, -1)   # shape [1, W]

    heatmaps = np.zeros((K, height, width), dtype=np.float32)

    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]

    for i, (kx, ky) in enumerate(keypoints):
        # pokud je keypoint mimo obrázek → prázdná mapa
        if kx < 0 or ky < 0 or kx >= width or ky >= height:
            continue

        # 2D Gaussian centered at (kx, ky)
        heatmap = np.exp(-((x - kx)**2 + (y - ky)**2) / (2 * sigma**2))
        heatmaps[i] = heatmap

    return heatmaps

