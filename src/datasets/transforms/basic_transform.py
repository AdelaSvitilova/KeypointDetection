import numpy as np
import albumentations as A


def resize(input_size, **kwargs):
    """
    Creates a resize augmentation.

    Args:
        input_size (tuple): Target size as (height, width).
        **kwargs: Additional unused keyword arguments for API compatibility.

    Returns:
        A.Resize: Albumentations resize transform.
    """
    return A.Resize(height=input_size[0], width=input_size[1], p=1.0)


def random_rotate(min, max, probability, **kwargs):
    """
    Creates a random rotation augmentation.

    Args:
        min (float): Minimum rotation angle (degrees).
        max (float): Maximum rotation angle (degrees).
        probability (float): Probability of applying the transform.
        **kwargs: Additional unused keyword arguments.

    Returns:
        A.Rotate: Albumentations rotation transform.
    """
    return A.Rotate(limit=(min, max), p=probability)


def affine(
    translate_percent_from,
    translate_percent_to,
    probability,
    rotate_min=0,
    rotate_max=0,
    **kwargs
):
    """
    Creates an affine transformation (translation + rotation).

    Args:
        translate_percent_from (float): Minimum translation ratio.
        translate_percent_to (float): Maximum translation ratio.
        probability (float): Probability of applying the transform.
        rotate_min (float): Minimum rotation angle.
        rotate_max (float): Maximum rotation angle.
        **kwargs: Additional unused keyword arguments.

    Returns:
        A.Affine: Albumentations affine transform.
    """
    return A.Affine(
        translate_percent={
            "x": (translate_percent_from, translate_percent_to),
            "y": (translate_percent_from, translate_percent_to),
        },
        rotate=(rotate_min, rotate_max),
        scale=1.0,
        p=probability,
    )


def flip(probability, **kwargs):
    """
    Creates a horizontal flip augmentation.

    Args:
        probability (float): Probability of applying the flip.
        **kwargs: Additional unused keyword arguments.

    Returns:
        A.HorizontalFlip: Albumentations flip transform.
    """
    return A.HorizontalFlip(p=probability)


def random_brightness(
    brightness_min,
    brightness_max,
    contrast_min,
    contrast_max,
    probability,
    **kwargs
):
    """
    Creates a random brightness and contrast augmentation.

    Args:
        brightness_min (float): Minimum brightness change.
        brightness_max (float): Maximum brightness change.
        contrast_min (float): Minimum contrast change.
        contrast_max (float): Maximum contrast change.
        probability (float): Probability of applying the transform.
        **kwargs: Additional unused keyword arguments.

    Returns:
        A.RandomBrightnessContrast: Albumentations brightness/contrast transform.
    """
    return A.RandomBrightnessContrast(
        brightness_limit=(brightness_min, brightness_max),
        contrast_limit=(contrast_min, contrast_max),
        p=probability,
    )


def clahe(clip_min, clip_max, tile_grid_min, tile_grid_max, probability, **kwargs):
    """
    Creates a CLAHE (Contrast Limited Adaptive Histogram Equalization) augmentation.

    Args:
        clip_min (float): Minimum clip limit.
        clip_max (float): Maximum clip limit.
        tile_grid_min (int): Minimum tile grid size.
        tile_grid_max (int): Maximum tile grid size.
        probability (float): Probability of applying the transform.
        **kwargs: Additional unused keyword arguments.

    Returns:
        A.CLAHE: Albumentations CLAHE transform.
    """
    return A.CLAHE(
        clip_limit=(clip_min, clip_max),
        tile_grid_size=(tile_grid_min, tile_grid_max),
        p=probability,
    )