import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

# (N, 3, 300, 300)
# x = torch.zeros((1, 3, 300, 300), dtype=torch.float)

vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
extra_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

def make_layers(vgg16_cfg, extra_cfg, batch_norm=False):
    '''
    :param vgg16_cfg: vgg16 layers
    :param extra_cfg: extra feature layers
    :param batch_norm: batch normailization option
    :return: vgg16 backbone + extra features layer ModulList
    '''
    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    in_channels = 1024
    for k, v in enumerate(extra_cfg):
        flag = False # 이것은 단지 그들이 깃발 변수가 무엇인지에 따라 커널 크기를 1 또는 3으로 설정하는 방법일 뿐이다.
        # (1,3)[False] -> 1
        # (1,3)[True] -> 3
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, extra_cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]), nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v

    return nn.ModuleList(layers)


class SSD300(nn.Module):
    def __init__(self, init_weights=True):
        super(SSD300, self).__init__()
        self.ssd = make_layers(vgg16_cfg, extra_cfg) # nn.ModuleList

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = [] # features to use
        for i in range(23):
            x = self.ssd[i](x)
        s = self.l2_norm(x)  # Conv4_3 feature
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, 35):
            x = self.ssd[i](x)
        features.append(x)

        for k, v in enumerate(self.ssd[35:]):
            if k in [3, 7, 11, 15]:
                features.append(x)

        return tuple(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# print(make_layers(vgg16_cfg, extra_cfg))
