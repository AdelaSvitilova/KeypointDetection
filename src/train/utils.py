import torch
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    """Custom collate function for a PyTorch DataLoader.

    This function merges a list of dataset samples into a single batch while
    safely handling fields that may contain `None`.

    Each dataset sample is expected to be a dictionary with the following keys:
        - "image": Image tensor or numpy array.
        - "keypoints": Tensor or numpy array containing keypoint coordinates.
        - "heatmaps": Tensor with heatmaps or None.
        - "filename": Name of the source image file (string).
        - "norm_coefficient": Normalization coefficient (float or tensor).

    Args:
        batch (list[dict]): List of samples returned by the dataset.

    Returns:
        dict: Dictionary where values are batched using PyTorch's
        `default_collate` when possible.

    Raises:
        ValueError: If a batch contains a mix of None and non-None values
            for the same key.
    """
    out = {}
    keys = batch[0].keys()  # Assume all samples share the same keys

    for key in keys:
        values = [item[key] for item in batch]

        # Case 1: all values are None -> keep None
        if all(v is None for v in values):
            out[key] = None

        # Case 2: mixture of None and non-None values -> explicit error
        elif any(v is None for v in values):
            raise ValueError(
                f"Mixed None and non-None values detected for key '{key}' in batch."
            )

        # Case 3: standard collation
        else:
            # For tensors this stacks them into a batch tensor.
            # For strings (e.g., filenames) it creates a list of strings.
            out[key] = default_collate(values)

    return out