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

def scale_keypoints(keypoints, orig_size, pred_size):
    """
    keypoints: (B, N, 2) or (B, N, 3)
    orig_size: [heights(B,), widths(B,)]
    pred_size: 
        - scalars [H_in, W_in]
        - array of shape (B, 2) e.g., [[H1, W1], [H2, W2]]
        - two separate arrays [heights(B,), widths(B,)]

    returns: (B, N, 2)
    """

    keypoints = np.asarray(keypoints)

    # np.atleast_1d ensures that even single scalar values act as 1D arrays
    H_orig = np.atleast_1d(orig_size[0])  # (B,) or (1,)
    W_orig = np.atleast_1d(orig_size[1])  # (B,) or (1,)

    # Safe conversion of the predicted size
    pred_size = np.atleast_1d(pred_size)

    # Unpack based on input shape
    if pred_size.ndim == 2 and pred_size.shape[1] == 2:
        # If input is in (B, 2) format, extract columns
        H_in = pred_size[:, 0]
        W_in = pred_size[:, 1]
    else:
        # If input is [H, W] (scalars) or two separate arrays [array_H, array_W]
        H_in = pred_size[0]
        W_in = pred_size[1]

    # Numpy automatically handles array/array and array/scalar division
    scale = np.stack(
        [W_orig / W_in, H_orig / H_in],
        axis=1
    )  # (B, 2) or (1, 2)

    return keypoints[:, :, :2] * scale[:, None, :]