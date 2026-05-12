import numpy as np
import albumentations as A

def resize(input_size, **kward):
    return A.Resize(height=input_size[0], width=input_size[1], p=1.0)

def random_rotate(min, max, probability, **kward):
    return A.Rotate(limit=(min, max), p=probability)

def affine(translate_percent_from, translate_percent_to, rotate_min, rotate_max, probability, **kward):
    return A.Affine(
        translate_percent={"x": (translate_percent_from, translate_percent_to), "y": (translate_percent_from, translate_percent_to)},
        rotate=(rotate_min, rotate_max),
        scale=1.0,
        p=probability
    )

def flip(probability, **kward):
    return A.HorizontalFlip(p=probability)

def random_brightness(brightness_min, brightness_max, contrast_min, contrast_max, probability, **kward):
    return A.RandomBrightnessContrast(
        brightness_limit=(brightness_min, brightness_max),
        contrast_limit=(contrast_min, contrast_max),
        p=probability
    )

def clahe(clip_min, clip_max, tile_grid_min, tile_grid_max, probability, **kward):
    return A.CLAHE(clip_limit=(clip_min, clip_max), tile_grid_size=(tile_grid_min, tile_grid_max), p=probability)