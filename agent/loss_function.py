import torch
from torch import nn
import torch.nn.functional as F
from utils.default_box_utils import hard_negative_mining

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes (smoothL1), and
    (2) a confidence loss for the predicted class scores. (softmax, cross-entropy)
    """
    def __init__(self, neg_pos_ratio=3, alpha=1.):
        '''
        :param neg_pos_ratio: number of positive samples : number of negative samples
        '''
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

    def forward(self, predicted_scores, predicted_locs, gt_labels, gt_locations):
        '''
        forward propagation

        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param gt_labels: true object labels, a list of N(batch size) tensors, (batch_size, num_priors)
        :param gt_locations: true object bounding boxes in boundary coordinates(cx, cy, w, h), a list of N tensors, (batch_size, num_priors, 4)
        :return: total loss = conf_loss + self.alpha * loc_loss
        '''
        # 8732 = prediction에 사용되는 피쳐맵들 크기 * 각 사용되는 default box개수 들의 합
        num_classes = predicted_scores.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(predicted_scores, dim=2)[:, :, 0] # 0 category(backgroud)'s log_softmax losses
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)
        # reduce the number of samples with hard negative mining
        predicted_scores = predicted_scores[mask, :] # mask가 2차원
        # hard negative mining 처리 완료된 predicted scores에서 crossentropy
        cls_loss = F.cross_entropy(predicted_scores.contiguous().view(-1, num_classes), gt_labels[mask], reduction='sum')

        pos_mask = gt_labels > 0
        # positive인 box들만(BG아닌 box들) 사용하는게 맞음
        predicted_locs = predicted_locs[pos_mask, :]
        gt_locations = gt_locations[pos_mask, :]
        loc_loss = F.smooth_l1_loss(predicted_locs.view(-1, 4), gt_locations.view(-1, 4), reduction='sum')
        num_pos = gt_locations.size(0)
        return (cls_loss + self.alpha * loc_loss)/num_pos

if __name__ == '__main__':
    p_s = torch.rand((2, 8732, 21))
    label = torch.randint(0, 21, (2, 8732))
    print(label)
    locs = torch.rand((2, 8732, 4))
    gt_locs = torch.rand((2, 8732, 4))
    m = MultiBoxLoss()
    m.forward(p_s, locs, label, gt_locs)