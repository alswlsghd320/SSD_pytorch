import os.path

#[Model]
name = 'SSD'
version = 0.0
update = 2021-5-14

#[Train]
HOME = os.path.expanduser("~")
VOC_ROOT = os.path.join(HOME, "/datasets/VOCdevkit/")
BACKBONE = 'VGG16'
VOC_CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
PRETRAINED = False
PRETRAIN_PATH = './dd.pt'
X_TRAIN_PATH=''
Y_TRAIN_PATH=''
X_VAL_PATH=''
Y_VAL_PATH=''
BATCH_SIZE=16
OPTIMIZER='Adam'
LR = 1e-3
EPOCHS=30
MOMENTUM=0.9
DECAY=0.0005
WIDTH=288
HEIGHT=512
PAD_WIDTH=288
PAD_HEIGHT=512
CROP_WIDTH=288
CROP_HEIGHT=512
SAVE_PATH=''
LOG_PATH='logs/'

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
