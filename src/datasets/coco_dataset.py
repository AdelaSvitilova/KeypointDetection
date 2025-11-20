import numpy as np
from pycocotools.coco import COCO
import cv2
import os
from .base_dataset import BaseDataset

class COCOKeypointDataset(BaseDataset):
    def __init__(self, root_dir, split='train2017', transform=None, num_keypoints=17):
        ann_dir = os.path.join(root_dir, 'annotations')
        ann_file = os.path.join(ann_dir, f'annotations_{split}.json')
        img_dir = os.path.join(root_dir, split)

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.num_keypoints = num_keypoints
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # normalize na 0-1

        orig_h, orig_w = image.shape[:2]
        target_size = (256, 256)
        image = cv2.resize(image, target_size)

        # Přepočet měřítka keypointů
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        if len(anns) > 0 and 'keypoints' in anns[0] and len(anns[0]['keypoints']) == self.num_keypoints * 3:
            kp = np.array(anns[0]['keypoints'], dtype=np.float32).reshape(self.num_keypoints, 3)
            kp[:, 0] *= scale_x
            kp[:, 1] *= scale_y
            keypoints = kp

        # Změna formátu obrazu na CxHxW (nebo HxWxC, podle toho, co chceš)
        image = image.transpose(2, 0, 1)  # CxHxW

        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        return image, keypoints
