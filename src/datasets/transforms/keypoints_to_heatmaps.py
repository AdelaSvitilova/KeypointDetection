import numpy as np

def keypoints_to_heatmaps_np(keypoints, size=(256, 256), sigma=2):
    """
    Converts keypoints into Gaussian heatmaps.

    This implementation is highly optimized:
    1. Computes the Gaussian function only in the local neighborhood of the point 
       (within a radius of 3 * sigma), which is significantly faster than computing 
       over the entire image.
    2. Correctly handles points that are partially out of bounds (e.g., due to 
       remove_invisible=False in Albumentations) – it renders their visible part 
       and safely clips the rest.

    Args:
        keypoints (np.ndarray): Array of coordinates with shape [K, 2] or [K, 3], 
                                where each row is (x, y) or (x, y, visibility).
        size (tuple): Output heatmap size as (height, width).
        sigma (float): Standard deviation of the Gaussian kernel (determines point size).

    Returns:
        np.ndarray: Heatmaps of shape [K, H, W], one heatmap per keypoint.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    K = keypoints.shape[0]
    height, width = size

    # Initialize zero heatmaps
    heatmaps = np.zeros((K, height, width), dtype=np.float32)

    # If keypoints include a visibility/confidence column (shape [K, 3]), drop it
    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]

    # Local window radius (3 * sigma covers 99.7% of the Gaussian distribution)
    radius = int(np.ceil(3 * sigma))

    for i, (kx, ky) in enumerate(keypoints):
        # Skip invalid keypoints (e.g., NaN values if a keypoint is missing in the dataset)
        if np.isnan(kx) or np.isnan(ky):
            continue

        # Define local bounding box around the keypoint
        # Min/max clamping ensures the bounding box does not exceed the heatmap boundaries
        x_min = max(0, int(np.floor(kx - radius)))
        x_max = min(width, int(np.ceil(kx + radius + 1)))
        y_min = max(0, int(np.floor(ky - radius)))
        y_max = min(height, int(np.ceil(ky + radius + 1)))

        # If the entire bounding box is completely outside the image, skip the keypoint
        if x_min >= x_max or y_min >= y_max:
            continue

        # Create a coordinate grid only for this local window
        y_grid = np.arange(y_min, y_max).reshape(-1, 1)
        x_grid = np.arange(x_min, x_max).reshape(1, -1)

        # Compute 2D Gaussian in the local window
        gaussian = np.exp(-((x_grid - kx) ** 2 + (y_grid - ky) ** 2) / (2 * sigma ** 2))

        # Insert the local Gaussian window into the overall heatmap
        heatmaps[i, y_min:y_max, x_min:x_max] = gaussian

    return heatmaps