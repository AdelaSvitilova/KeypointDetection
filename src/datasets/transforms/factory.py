from .basic_transform import random_rotate, random_brightness, flip, clahe, resize, affine
import albumentations as A

def get_transform(
    transform, format, input_size
):
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

    transforms_list.append(resize(input_size))
    
    if transform:
        for name, params in transform.items():
            if name not in transforms:
                raise ValueError(f"Unknown transform: {name}")

            if params["use"] == True:
                transform_class = transforms[name]
                transforms_list.append(transform_class(**params))

    compose_kwargs = {}

    if format == "heatmaps":
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