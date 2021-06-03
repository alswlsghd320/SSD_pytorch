import torch
from torch import nn
import torch.nn.functional as F
from utils.default_box_utils import hard_negative_mining, cxcy_to_gcxgcy, DefaultBox

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes (smoothL1), and
    (2) a confidence loss for the predicted class scores. (softmax, cross-entropy)
    """
    def __init__(self, neg_pos_ratio=3, alpha=1., default_box=DefaultBox(), device='cpu'):
        '''
        :param neg_pos_ratio: number of positive samples : number of negative samples
        '''
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.default_box = default_box().to(device, dtype=torch.float32)
        self.device = device

    def forward(self, conf_pred, loc_pred, conf_true, loc_true):
        '''
        forward propagation
        :param conf_pred: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param loc_pred: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param conf_true: true object labels, a list of N(batch size) tensors, (batch_size, num_defaults)
        :param loc_true: true object bounding boxes in boundary coordinates(cx, cy, w, h), a list of N tensors, (batch_size, num_defaults, 4)
        :return: total loss = conf_loss + self.alpha * loc_loss
        '''
        # 8732 = prediction에 사용되는 피쳐맵들 크기 * 각 사용되는 default box개수 들의 합
        num_classes = conf_pred.size(2)

        # Confidence loss
        with torch.no_grad():
            loss = -F.log_softmax(conf_pred, dim=2)[:, :, 0] # 0 category(backgroud)'s log_softmax losses
            mask = hard_negative_mining(loss, conf_true, self.neg_pos_ratio, device=self.device) # (batch_size, num_defaults)
        # reduce the number of samples with hard negative mining
        conf_pred = conf_pred[mask, :] # mask가 2차원
        # hard negative mining 처리 완료된 predicted scores에서 crossentropy
        cls_loss = F.cross_entropy(conf_pred.contiguous().view(-1, num_classes), conf_true[mask], reduction='sum')

        # Localization loss
        pos_mask = conf_true > 0
        # positive인 box들만(BG아닌 box들) 사용하는게 맞음
        g_loc_true = loc_true[pos_mask, :]

        loc_loss = F.smooth_l1_loss(g_loc_true.view(-1, 4), loc_pred[pos_mask].view(-1, 4), reduction='sum')

        num_pos = pos_mask.long().sum() #positive 개수의 총합

        return cls_loss/num_pos, self.alpha * loc_loss / num_pos

