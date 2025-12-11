from .random_dataset import RandomDataset
from .random_dataset_heatmaps import RandomHeatmapDataset
from .coco_dataset import COCOKeypointDataset
from .coco_dataset_new import COCODataset

from .transforms import KeypointsToHeatmaps

def get_dataset(name, load=None, **kwargs):
    datasets = {
        "random": RandomDataset,
        "random_heatmaps": RandomHeatmapDataset,
        "coco": COCOKeypointDataset,
        "coco_new": COCODataset,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    DatasetClass = datasets[name]

    if load is not None:
        return DatasetClass(load=load,**kwargs)
    
    return DatasetClass(**kwargs)
