from configs import ssd300 as cfg
import os
import cv2

import xml.etree.ElementTree as ET
import pathlib
import numpy as np
from PIL import Image
import torch


class VOCDataset():

    def __init__(self, root, is_test=False, transform=None, target_transform=None, keep_difficult=False):
        self.root = pathlib.Path(cfg.VOC_ROOT)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = f"{self.root}/ImageSets/Main/test.txt"
        else:
            image_sets_file = f"{self.root}/ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(cfg.VOC_CLASSES)}

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
        return image, boxes, labels  # image(RGB), boxes = [x1,y1,x2,y2], background label is 0

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
            # we're only concerned with classes in our list
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

        return (np.array(boxes, dtype=np.float32),  # (object 수, 4) x1,y1,x2,y2
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def _read_image(self, image_id):
        image_file = f"{self.root}/JPEGImages/{image_id}.jpg"
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image


if __name__ is '__main__':
    train_dataset = VOCDataset(cfg.VOC_ROOT)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.BATCH_SIZE,
                                               shuffle=True)