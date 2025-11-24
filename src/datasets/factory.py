from .random_dataset import RandomDataset
from .random_dataset_heatmaps import RandomHeatmapDataset
from .coco_dataset import COCOKeypointDataset

def get_dataset(name, split=None, **kwargs):
    datasets = {
        "random": RandomDataset,
        "random_heatmaps": RandomHeatmapDataset,
        "coco": COCOKeypointDataset,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    DatasetClass = datasets[name]

    # pokud split je None → načteme "vše"
    if split is None:
        return DatasetClass(**kwargs)  # třída sama načte všechny obrázky ve složce

    # pokud je split list → projdeme jej
    loaded_datasets = []
    for s in split:
        ds_kwargs = kwargs.copy()
        ds_kwargs.pop("split", None)  # odstraníme split, dataset ho nezná
        loaded_datasets.append(DatasetClass(split=s, **ds_kwargs))

    if len(loaded_datasets) == 1:
        return loaded_datasets[0]
    return tuple(loaded_datasets)
