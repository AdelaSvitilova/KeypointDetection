import numpy as np
from pycocotools.coco import COCO
import cv2
import os
import csv
from .base_dataset import BaseDataset

class COCODataset(BaseDataset):
    def __init__(self, root_dir, load, transform=None, num_keypoints=17, target_size=(256,256)):
        ann_file = os.path.join(root_dir, 'annotations.json')
        self.img_dir = os.path.join(root_dir, 'images')
        load_file = os.path.join(root_dir, load)

        self.coco = COCO(ann_file)
        self.transform = transform
        self.num_keypoints = num_keypoints
        self.target_size = target_size

        # Načti názvy souborů
        with open(load_file, newline='') as f:
            file_names = [row[0].strip() for row in csv.reader(f) if row]

        # Mapování file_name → img_id
        name_to_id = {
            info["file_name"]: img_id
            for img_id, info in self.coco.imgs.items()
        }

        # === Dvě hlavní pole ===
        self.images = []
        self.keypoints = []

        for fname in file_names:
            if fname not in name_to_id:
                continue

            img_id = name_to_id[fname]

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            kp = np.zeros((num_keypoints, 3), dtype=np.float32)

            if anns and "keypoints" in anns[0]:
                raw = anns[0]["keypoints"]
                if len(raw) == num_keypoints * 3:
                    kp = np.array(raw, dtype=np.float32).reshape(num_keypoints, 3)

            self.images.append(fname)
            self.keypoints.append(kp)

        print(load, self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # načti obrázek
        path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        orig_h, orig_w = image.shape[:2]

        # resize
        image = cv2.resize(image, self.target_size)

        # scale pro keypointy
        sx = self.target_size[0] / orig_w
        sy = self.target_size[1] / orig_h

        keypoints = self.keypoints[idx].copy()
        keypoints[:, 0] *= sx
        keypoints[:, 1] *= sy

        # CHW
        image = image.transpose(2, 0, 1)

        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        return image, keypoints
