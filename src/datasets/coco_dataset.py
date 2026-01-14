import numpy as np
from pycocotools.coco import COCO
import cv2
import os
import csv
from .base_dataset import BaseDataset


class COCODataset(BaseDataset):
    """COCO dataset loader for keypoint detection.

    This dataset loads images and keypoints from the COCO dataset
    and optionally converts keypoints to heatmaps. It supports
    resizing images and scaling keypoints to a desired output resolution.

    Args:
        root_dir (str): Path to the dataset root containing 'images' and 'annotations.json'.
        load (str): CSV file listing image filenames to load.
        num_samples (int, optional): Maximum number of samples to load.
        transform (callable, optional): Optional transform applied to images and keypoints.
        heatmaps (callable, optional): Function to convert keypoints to heatmaps.
        num_keypoints (int): Number of keypoints per sample.
        input_size (tuple): Spatial resolution for input images (height, width).
        output_size (tuple): Spatial resolution for output keypoints/heatmaps (height, width).
    """

    def __init__(
        self,
        root_dir,
        load,
        num_samples=None,
        transform=None,
        heatmaps=None,
        num_keypoints=17,
        input_size=(256, 256),
        output_size=(64, 64),
    ):
        # Paths to annotations, images, and CSV load file
        ann_file = os.path.join(root_dir, 'annotations.json')
        self.img_dir = os.path.join(root_dir, 'images')
        load_file = os.path.join(root_dir, load)

        # Initialize COCO API
        self.coco = COCO(ann_file)
        self.transform = transform
        self.to_heatmaps = heatmaps
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_size = output_size

        # Read list of filenames from CSV
        with open(load_file, newline='') as f:
            file_names = [row[0].strip() for row in csv.reader(f) if row]

        # Map filename -> COCO image ID
        name_to_id = {info["file_name"]: img_id for img_id, info in self.coco.imgs.items()}

        # Lists to store loaded images and keypoints
        self.images = []
        self.keypoints = []

        loaded = 0
        max_samples = num_samples if num_samples is not None else float("inf")

        # Load keypoints for each file
        for fname in file_names:
            if loaded >= max_samples:
                break

            if fname not in name_to_id:
                continue

            img_id = name_to_id[fname]

            # Get annotations for the image
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # Skip images without keypoints
            if not anns or "keypoints" not in anns[0]:
                continue

            raw = anns[0]["keypoints"]

            # Ensure keypoints length matches expected number
            if len(raw) != num_keypoints * 3:
                continue

            # Convert to numpy array (num_keypoints, 3)
            kp = np.array(raw, dtype=np.float32).reshape(num_keypoints, 3)

            self.images.append(fname)
            self.keypoints.append(kp)

            loaded += 1

    def __len__(self):
        """Return the total number of loaded samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieve a single sample: image, keypoints, and optional heatmaps.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing:
                - image (np.ndarray): Image tensor in CHW format.
                - keypoints (np.ndarray): Keypoints array (num_keypoints, 3).
                - heatmaps (np.ndarray or None): Heatmaps if heatmap function is provided.
        """
        # Load image and convert to RGB, normalize to [0,1]
        path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        orig_h, orig_w = image.shape[:2]

        # Resize image to input size
        image = cv2.resize(image, self.input_size)

        # Compute scaling factors for keypoints
        sx = self.output_size[0] / orig_w
        sy = self.output_size[1] / orig_h

        # Copy keypoints and scale to output size
        keypoints = self.keypoints[idx].copy()
        keypoints[:, 0] *= sx
        keypoints[:, 1] *= sy

        # Convert image to CHW format
        image = image.transpose(2, 0, 1)

        # Apply optional transform
        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        # Convert keypoints to heatmaps if requested
        heatmaps = None
        if self.to_heatmaps:
            heatmaps = self.to_heatmaps(keypoints, self.output_size)

        return {
            "image": image,
            "heatmaps": heatmaps,
            "keypoints": keypoints,
        }
