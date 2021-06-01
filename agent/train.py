import torch
from torch.utils.data import DataLoader

import os
import tqdm
from datetime import datetime as dt

from configs import ssd300 as cfg
from datasets.dataset import VOCDataset
from models.ssd300 import SSD300
from agent.loss_function import MultiBoxLoss
from utils.data_augmentation import SSDAugmentation
from utils.utils import get_optimizer
from utils.matching_strategy import get_matching_label

def train():


    # Define Dataset
    train_ds = VOCDataset(root=cfg.VOC_ROOT, transform=SSDAugmentation(size=cfg.SIZE, mean = cfg.MEANS) )

    # Define Model
    num_classes = 1 if len(cfg.VOC_CLASSES)==1 else len(cfg.VOC_CLASSES)+1
    net = SSD300(init_weights=False, num_classes=num_classes)

    if torch.cuda.is_available() and cfg.is_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net.cuda()

    if cfg.PRETRAINED:
        weights = torch.load(cfg.PRETRAINED_PATH)
        net.load_state_dict(weights)

    optimizer = get_optimizer(net, cfg.OPTIMIZER, lr=cfg.LR, momentum=cfg.MOMENTUM, decay=cfg.DECAY)
    criterion = MultiBoxLoss()

    net.train()

    loss_loc = []
    loss_conf = []

    train_dl = DataLoader(train_ds,
                          cfg.BATCH_SIZE,
                          num_workers=cfg.NUM_WORKERS,
                          shuffle=True,
                          collate_fn=train_ds.collate_fn,
                          pin_memory=True)

    for epoc in range(cfg.EPOCHS):
        running_loc_loss = 0.0
        running_conf_loss = 0.0

        for i, (img, loc_t, conf_t) in tqdm.tqdm(enumerate(train_dl), total=len(train_dl), mininterval=0.01):
            loc_pred, conf_pred = net(img)
            loc_true, conf_true = get_matching_label(loc_t, conf_t, criterion.default_box, cfg.threshold)

            optimizer.zero_grad()

            loss_c, loss_l = criterion(conf_pred, loc_pred, conf_true, loc_true)
            loss = loss_c + loss_l
            loss.backward()

            optimizer.step()

            running_loc_loss += loss_l.item()
            running_conf_loss += loss_c.item()

            if i % 10 == 0:
                print(f"loss : {loss.item():.4f}")

        loss_loc.append(running_loc_loss)
        loss_conf.append(running_conf_loss)

    save_path = cfg.SAVE_PATH
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, f'SSD_{dt.today().strftime("%Y%m%d%H%M")}.pth'))

