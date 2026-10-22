from .basic_transform import random_rotate, random_brightness, flip, clahe, resize, affine, random_crop
import albumentations as A

def get_transform(
    transform, format, input_size, window_size, predict_type=None, **kwargs
):
    """Composes a list of Albumentations transforms based on configuration.

    Args:
        transform: A dictionary containing transform names as keys and their
            parameters (including a 'use' boolean flag) as values.
        format: A string specifying the dataset format (e.g., 'heatmaps').
        input_size: A tuple or integer representing the target size for the
            initial resize operation.

    Returns:
        An albumentations.Compose object initialized with the selected
        transforms and format-specific configurations.

    Raises:
        ValueError: If an unknown transform name is provided in the configuration.
    """
    print("transforms: ", transform)
    
    # Mapping from dataset name to dataset class
    transforms = {
        "rotate": random_rotate,
        "translate": affine,
        "brightness": random_brightness,
        "flip": flip,
        "clahe": clahe,
    }

    transforms_list = []

    if predict_type == "sliding_window" or predict_type == "sliding_window_val":
        pass
    else:
        transforms_list.append(resize(input_size))
    
    if transform:
        for name, params in transform.items():
            if name not in transforms:
                raise ValueError(f"Unknown transform: {name}")

            # Dynamic unpacking of parameters if the transform is enabled
            if params["use"] == True:
                transform_class = transforms[name]
                transforms_list.append(transform_class(**params))

    if predict_type == "sliding_window":
        transforms_list.append(random_crop(window_size))

    compose_kwargs = {}

    
    compose_kwargs["keypoint_params"] = (
        A.KeypointParams(
            format="xy",
            remove_invisible=False,
        )
    )

    return A.Compose(
        transforms_list,
        **compose_kwargs,
    )