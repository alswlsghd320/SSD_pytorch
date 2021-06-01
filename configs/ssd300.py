import os

#[Model]
name = 'SSD'
version = 0.0
update = 2021-5-14

#[Train]
HOME = os.path.abspath(".")
VOC_ROOT = os.path.join(HOME, "datasets/VOCdevkit/VOC2007")
BACKBONE = 'VGG16'
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
PRETRAINED = False
PRETRAINED_PATH = '.pt'
is_cuda = True
BATCH_SIZE=16
OPTIMIZER='Adam'
LR = 1e-3
EPOCHS=2
MOMENTUM=0.9
DECAY=0.0005
NUM_WORKERS=8
SIZE=300
MEANS = (104, 117, 123) # average of BGR order in Imagenet
SAVE_PATH = None
LOG_PATH = 'logs/'
threshold = 0.5
FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
AR_STEPS = [4, 6, 6, 6, 4, 4]
ASPECT_RATIOS = [1, 2, 0.5, 3, 1 / 3]
SK_MIN = 0.2
SK_MAX = 0.9

#[Test]
BACKBONE = 'VGG16'
BATCH_SIZE=16
LR = 1e-3
MOMENTUM=0.9
DECAY=0.0005
WIDTH=288
HEIGHT=512
PAD_WIDTH=288
PAD_HEIGHT=512
CROP_WIDTH=288
CROP_HEIGHT=512
