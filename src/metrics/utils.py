import numpy as np

def heatmaps_to_keypoints(heatmaps):
    """
    heatmaps: np.ndarray of shape (B, K, H, W)
    returns:  np.ndarray of shape (B, K, 3)  -> (x, y, visible)
    """
    B, K, H, W = heatmaps.shape

    # 1. zploštění H×W
    heatmaps_flat = heatmaps.reshape(B, K, -1)  # (B, K, H*W)

    # 2. index maxima
    idx = np.argmax(heatmaps_flat, axis=2)  # (B, K)

    # 3. převod na (x, y)
    y = idx // W
    x = idx % W

    # 4. visible flag (všude 1)
    visible = np.ones((B, K), dtype=heatmaps.dtype)

    # 5. složení (x, y, visible)
    keypoints = np.stack([x, y, visible], axis=2)  # (B, K, 3)

    return keypoints
