from .coco_dataset import COCODataset
from .random_datset import RandomDataset

from .transforms.keypoints_to_heatmaps import keypoints_to_heatmaps_np

def get_dataset(name, load=None, keypoint_format=None, **kwargs):
    datasets = {
        "coco": COCODataset,
        "random": RandomDataset
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    DatasetClass = datasets[name]

    heatmaps = None
    
    if keypoint_format == "heatmaps":
        heatmaps = keypoints_to_heatmaps_np

    if load is not None:
        return DatasetClass(load=load, heatmaps=heatmaps, **kwargs)
    
    return DatasetClass(heatmaps=heatmaps, **kwargs)
