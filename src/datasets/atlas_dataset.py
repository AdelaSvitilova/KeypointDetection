import numpy as np
import cv2
import os
import csv
from .base_dataset import BaseDataset


class AtlasDataset(BaseDataset):
    """
    Dataset loader for keypoint detection using CSV annotations (x, y).

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
        annotation_file='pose_dataset.csv',
    ):
        self.img_dir = os.path.join(root_dir, "images")
        ann_file = os.path.join(root_dir, annotation_file)
        load_file = os.path.join(root_dir, load)

        self.transform = transform
        self.to_heatmaps = heatmaps
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_size = output_size

        # načti seznam obrázků, které chceš použít
        with open(load_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # přeskočí hlavičku
            valid_files = {row[0].strip() for row in reader if row}

        self.images = []
        self.keypoints = []

        max_samples = num_samples if num_samples is not None else float("inf")
        loaded = 0

        # načti anotace
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
                    continue  # špatný počet keypointů

                kp = coords.reshape(num_keypoints, 2)

                self.images.append(fname)
                self.keypoints.append(kp)
                loaded += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        orig_h, orig_w = image.shape[:2]

        # resize image
        image = cv2.resize(image, self.input_size)

        # scale keypoints
        sx = self.output_size[0] / orig_w
        sy = self.output_size[1] / orig_h

        keypoints = self.keypoints[idx].copy()
        keypoints[:, 0] *= sx
        keypoints[:, 1] *= sy

        # CHW
        image = image.transpose(2, 0, 1)

        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        heatmaps = None
        if self.to_heatmaps:
            heatmaps = self.to_heatmaps(keypoints, self.output_size)

        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "filename": self.images[idx],
        }
