from .coco_dataset import COCODataset
from .random_dataset import RandomDataset

from .transforms.keypoints_to_heatmaps import keypoints_to_heatmaps_np

def get_dataset(name, load=None, num_samples=None, keypoint_format=None, **kwargs):
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
        return DatasetClass(load=load, num_samples=num_samples, heatmaps=heatmaps, **kwargs)
    
    return DatasetClass(num_samples=num_samples, heatmaps=heatmaps, **kwargs)
