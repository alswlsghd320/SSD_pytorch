import os

import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch

from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as standard_transforms

import utils.voc_utils as extended_transforms
from utils.voc_utils import make_dataset


class VOC(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.imgs)


class VOCDataLoader(DataLoader):
    def __init__(self):
        pass

    def finalize(self):
        pass