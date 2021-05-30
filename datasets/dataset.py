import os

import xml.etree.ElementTree as ET
import pathlib
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as standard_transforms

class VOCDataset(torch.utils.data.Dataset):

    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root, is_test=False, transform=None, target_transform=None, keep_difficult=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root # "ImageSets/Main/test.txt"
        else:
            image_sets_file = f"{self.root}/ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}


    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels
        

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)


    
    def _get_annotation(self, image_id):
        annotation_file = f"{self.root}/Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self,image_id):
        image_file = f"{self.root}/JPEGImages/{image_id}.jpg"
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

if __name__ is '__main__':
    train_dataset = VOCDataset('')

