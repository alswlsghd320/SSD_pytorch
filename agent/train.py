import torch
import os

from datasets.dataset import VOCDataset
from models.ssd300 import SSD300

def train(cfg=None):
    train_path = cfg['X_TRAIN_PATH']
    val_path = cfg['X_VAL_PATH']

    num_classes = cfg.getint('num_classes')
    batch_size = cfg.getint('batch_size')
    epochs = cfg.getint('epochs')
    lr = cfg.getfloat('LR')

    # Define Dataset
    train_ds = VOCDataset(train_path)

    # Define Model
    net = SSD300(init_weights=True, num_classes=num_classes)

    if torch.cuda.is_available() and cfg.getboolean('is_cuda'):
        net.cuda()

