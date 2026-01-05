import torch
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    """
    batch: list of dicts, každý dict má klíče:
        - 'image': tensor / np.ndarray
        - 'heatmaps': tensor / None
        - 'keypoints': tensor / np.ndarray
    """
    out = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        # Pokud všechny hodnoty jsou None → nech None
        if all(v is None for v in values):
            out[key] = None
        # Pokud jsou mix None a tensor → chyba, aby ses nespálila
        elif any(v is None for v in values):
            raise ValueError(f"Mixed None / non-None values for key '{key}' in batch")
        # Jinak → použij default_collate (stackne tensory)
        else:
            out[key] = default_collate(values)

    return out
