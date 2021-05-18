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
        flag = False # flag가 무엇인지에 따라 커널 크기를 1 또는 3으로 설정하는 방법
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
    def __init__(self, init_weights=True, num_class=20):
        super(SSD300, self).__init__()
        self.ssd = make_layers(vgg16_cfg, extra_cfg) # nn.ModuleList
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.out_channels = [] # out channels of conf4_3, 7, 8_2, 9_2, 10_2, 11_2
        self.loc_conv = [] # for bbox regression
        self.cls_conv = [] # for classification
        self.l2_norm = L2Norm(512, scale=20)
        for i in [22, 34, 38, 42, 46, 50]:
            self.out_channels.append(self.ssd[i].out_channels)

        for nd, oc in zip(self.num_defaults, self.out_channels):
            self.loc_conv.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.cls_conv.append(nn.Conv2d(oc, nd * num_class, kernel_size=3, padding=1))

        self.loc_conv = nn.ModuleList(self.loc)
        self.cls_conv = nn.ModuleList(self.conf)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = [] # features to use
        for i in range(23):
            x = self.ssd[i](x)
        s = x
        # conv4_3 L2 norm
        norm = s.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        s = s / norm  # (N, 512, 38, 38)
        s = s * self.rescale_factors  # (N, 512, 38, 38) out_ch=512
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, 35):
            x = self.ssd[i](x)
        features.append(x) # Conv7 feature out_ch=1024

        # for k, v in enumerate(self.ssd[35:]):
        #     if k in [3, 7, 11, 15]:
        #         features.append(x)
        # 아앗.. batch norm 할 경우 생각 x.... 나중에 고쳐볼게요...
        for i in range(35, 51):
            if i in [38, 42, 46, 50]:
                x = self.ssd[i](x)
                features.append(x) # out_ch 512, 256, 256, 256

        return tuple(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # = He Initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

print(make_layers(vgg16_cfg, extra_cfg)[5].out_channels)
