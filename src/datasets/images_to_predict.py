import cv2
import os
import csv
import numpy as np
from .base_dataset import BaseDataset


class ImageDataset(BaseDataset):
    """
    Simple image-only dataset for inference / prediction.

    Loads images listed in a CSV file and applies optional transforms.
    """

    def __init__(
        self,
        root_dir,
        load,
        num_samples=None,
        transform=None,
        input_size=(256, 256),
        heatmaps=None,
        annotation_file=None
    ):
        self.img_dir = os.path.join(root_dir, "images")
        load_file = os.path.join(root_dir, load)

        self.transform = transform
        self.input_size = input_size

        # Read filenames from CSV
        with open(load_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # přeskočí první řádek (hlavičku)
            file_names = [row[0].strip() for row in csv.reader(f) if row]

        if num_samples is not None:
            file_names = file_names[:num_samples]

        self.images = file_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        path = os.path.join(self.img_dir, fname)

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Resize
        image = cv2.resize(image, self.input_size)

        # CHW
        image = image.transpose(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "filename": self.images[idx],
        }
