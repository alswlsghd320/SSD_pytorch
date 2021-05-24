import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
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
    flag = False
    for k, v in enumerate(extra_cfg):
        # flag가 무엇인지에 따라 커널 크기를 1 또는 3으로 설정하는 방법
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
    def __init__(self, init_weights=True, num_class=21):
        super(SSD300, self).__init__()
        self.ssd = make_layers(vgg16_cfg, extra_cfg) # nn.ModuleList
        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        # rescale factor은 학습된다 -> nn.Parameter
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.out_channels = [] # out channels of conf4_3, 7, 8_2, 9_2, 10_2, 11_2
        self.loc_conv = [] # for bbox regression
        self.cls_conv = [] # for classification
        self.num_class = num_class
        self.out_channels = [512, 1024, 512, 256, 256, 256]
        for nd, oc in zip(self.num_defaults, self.out_channels):
            self.loc_conv.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.cls_conv.append(nn.Conv2d(oc, nd * num_class, kernel_size=3, padding=1))

        self.loc_conv = nn.ModuleList(self.loc_conv)
        self.cls_conv = nn.ModuleList(self.cls_conv)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feat4_3_range = 23
        feat7_range = 35
        if len(self.ssd) == 64:
            feat4_3_range = 33
            feat7_range = 48

        features = [] # features to use
        for i in range(feat4_3_range):
            x = self.ssd[i](x)
        s = x
        # conv4_3 L2 norm
        norm = s.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        s = s / norm  # (N, 512, 38, 38)
        s = s * self.rescale_factors  # (N, 512, 38, 38) out_ch=512
        features.append(s)

        # apply vgg up to fc7
        for i in range(feat4_3_range, feat7_range):
            x = self.ssd[i](x)
        features.append(x) # Conv7 feature out_ch=1024

        feat8_2_idx = feat7_range + 3
        for i in range(feat7_range, len(self.ssd)):
            x = self.ssd[i](x)
            if i in [feat8_2_idx, feat8_2_idx+4, feat8_2_idx+8, feat8_2_idx+12]:
                features.append(x) # out_ch 512, 256, 256, 256

        classes_scores= []
        batch_size = self.cls_conv[0](features[0]).size(0)
        for i, feature in enumerate(features):
            f = self.cls_conv[i](feature)
            f = f.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
            f = f.view(batch_size, -1, self.num_class)  # (N, 5776, 4), there are a total 5776 boxes on this feature map
            classes_scores.append(f)

        locs = []
        for i, feature in enumerate(features):
            l = self.loc_conv[i](feature)  # (N, 16, 38, 38)
            l = l.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
            l = l.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map
            locs.append(l)

        return locs, classes_scores

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # = He Initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# if __name__ == '__main__':
#     # (N, 3, 300, 300)
#     x = torch.zeros((1, 3, 300, 300), dtype=torch.float)
#
#     print(make_layers(vgg16_cfg, extra_cfg, batch_norm=True))
#     ssd = SSD300()
#     ssd.forward(x)
