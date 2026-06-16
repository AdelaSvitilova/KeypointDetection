from .coco_dataset import COCODataset
from .random_dataset import RandomDataset
from .images_to_predict import ImageDataset
from .atlas_dataset import AtlasDataset
from .transforms.keypoints_to_heatmaps import keypoints_to_heatmaps_np


def get_dataset(
    name,
    load=None,
    num_samples=None,
    keypoint_format=None,
    **kwargs,
):
    """Factory function for creating dataset instances.

    This function instantiates a dataset based on its name and configuration.
    It supports optional conversion of keypoints into heatmaps and forwards
    additional keyword arguments to the dataset constructor.

    Args:
        name (str): Name of the dataset to create. Supported values are
            "coco" and "random".
        load (str, optional): Path or identifier used to load dataset data.
            Only applicable to datasets that support loading from disk.
        num_samples (int, optional): Number of samples to use from the dataset.
        keypoint_format (str, optional): Format of keypoint representation.
            If set to "heatmaps", keypoints will be converted to heatmaps.
        **kwargs: Additional keyword arguments passed to the dataset
            constructor.

    Returns:
        BaseDataset: An instance of the requested dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    # Mapping from dataset name to dataset class
    datasets = {
        "coco": COCODataset,
        "random": RandomDataset,
        "images": ImageDataset,
        "atlas": AtlasDataset,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    DatasetClass = datasets[name]

    # Optional keypoints-to-heatmaps conversion function
    heatmaps = None
    if keypoint_format == "heatmaps":
        heatmaps = keypoints_to_heatmaps_np

    # Instantiate dataset with or without loading argument
    if load is not None:
        return DatasetClass(
            load=load,
            num_samples=num_samples,
            heatmaps=heatmaps,
            **kwargs,
        )

    return DatasetClass(
        num_samples=num_samples,
        heatmaps=heatmaps,
        **kwargs,
    )
