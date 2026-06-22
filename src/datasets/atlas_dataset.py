import numpy as np
import cv2
import os
import csv
from .base_dataset import BaseDataset

class AtlasDataset(BaseDataset):
    """
    Dataset loader for keypoint detection from CSV annotations (x, y).

    CSV format:
        filename, x1, y1, x2, y2, ..., xN, yN
    """

    def __init__(
        self,
        root_dir,
        load,
        num_samples=None,
        transform=None,
        heatmaps=None,
        num_keypoints=23,
        input_size=(256, 256),
        output_size=(64, 64),
        annotation_file="pose_dataset.csv",
        norm_coefficient=25,
        **kwargs
    ):
        """
        Args:
            root_dir: Root directory with subfolders images/ and annotations.
            load: Path to CSV file listing image filenames to use.
            num_samples: Maximum number of samples to load; None means all.
            transform: Optional transform to apply to image and keypoints.
            heatmaps: Callable that converts keypoints to heatmaps, if any.
            num_keypoints: Number of keypoints per instance.
            input_size: (H, W) to which images are resized before feeding to the model.
            output_size: (H, W) of the heatmap/feature space.
            annotation_file: CSV file containing keypoint coordinates.
            norm_coefficient: Normalization factor for metrics like PCK.
        """
        self.img_dir = os.path.join(root_dir, "images")
        ann_file = os.path.join(root_dir, annotation_file)
        load_file = os.path.join(root_dir, load)

        self.transform = transform
        self.to_heatmaps = heatmaps
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_size = output_size
        self.norm_coefficient = norm_coefficient

        # Load list of image filenames to use.
        with open(load_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header.
            valid_files = {row[0].strip() for row in reader if row}

        self.images = []
        self.keypoints = []

        max_samples = num_samples if num_samples is not None else float("inf")
        loaded = 0

        # Load annotations.
        with open(ann_file, newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if loaded >= max_samples:
                    break

                fname = row[0].strip()
                if fname not in valid_files:
                    continue

                coords = np.array(row[1:], dtype=np.float32)

                if coords.size != num_keypoints * 2:
                    continue  # Skip invalid keypoint count.

                kp = coords.reshape(num_keypoints, 2)

                self.images.append(fname)
                self.keypoints.append(kp)
                loaded += 1

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return a single sample as dict."""
        path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        keypoints = self.keypoints[idx].copy()

        img_size = image.shape
        orig_h, orig_w = self.input_size
        out_h, out_w = self.output_size

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        # Scale keypoints to output size.
        sx = out_w / orig_w
        sy = out_h / orig_h

        keypoints[:, 0] *= sx
        keypoints[:, 1] *= sy

        # Convert to CHW.
        image = (image / 255.0).astype(np.float32)
        image = image.transpose(2, 0, 1)

        heatmaps = None
        if self.to_heatmaps:
            heatmaps = self.to_heatmaps(keypoints, self.output_size)

        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "filename": self.images[idx],
            "norm_coefficient": self.norm_coefficient,
            "orig_height": img_size[0], 
            "orig_width": img_size[1],
            #"orig_size": img_size
        }