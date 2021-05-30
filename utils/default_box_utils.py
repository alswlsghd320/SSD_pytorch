
from numpy import sqrt
import torch
from itertools import product as product
import math
def cxcy_to_xyxy(cxcy):
    """
    Convert bbox from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)

    :param cxcy:bbox coordinates having center-size coordinates, a tensor of size(n_boxes, 4)
    :return: bbox in (xmin, ymin, xmax, ymax) coordinates
    """
    return torch.cat([cxcy[..., :2] - (cxcy[..., 2:]/2),  cxcy[..., :2] + (cxcy[..., 2:]/2)], -1)

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy (n_priors, 4): bounding boxes in center-size coordinates
    :param priors_cxcy (n_priors, 4): prior boxes with respect to which the encoding must be performed
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:]),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:])], 1)  # g_w, g_h


def iou(gt_boxes, predicted_boxes):
    """Return intersection-over-union (Jaccard index) of boxes.
        each box has (cx, cy, w, h) coordinates

    Args:
        gt_boxes (n1, 4): ground truth boxes.
        predicted_boxes (n2, 4): predicted boxes.

    Returns:
        iou (N): IoU values.
    """
    xyxy_gt_boxes = cxcy_to_xyxy(gt_boxes)
    xyxy_predicted_boxes = cxcy_to_xyxy(predicted_boxes)
    overlap_left_top = torch.max(xyxy_gt_boxes[..., :2], xyxy_predicted_boxes[..., :2])
    overlap_right_bottom = torch.min(xyxy_gt_boxes[..., 2:], xyxy_predicted_boxes[..., 2:])
    overlap_area = (overlap_right_bottom[..., 0] - overlap_left_top[..., 0]) * \
                   (overlap_right_bottom[..., 1] - overlap_left_top[..., 1])

    gt_area = gt_boxes[..., 2] * gt_boxes[..., 3]
    pred_area = predicted_boxes[..., 2] * predicted_boxes[..., 3]
    return overlap_area / (gt_area + pred_area - overlap_area + 1e-5)


def assign_priors(gt_boxes, gt_labels, priors_boxes, iou_threshold=0.5):
    """
    Assign ground truth boxes and targets to priors
    change labels of negative boxes to 0(BG label)

    :param gt_boxes (num_targets, 4):ground truth boxes
    :param gt_labels(num_targets):
    :param priors_boxes (num_priors, 4): priors' coordinates (cx, cy, w, h)
    :return:
        boxes(num_priors, 4) : real values for priors
        labels(num_priors): labels for priors
    """
    ious = iou(gt_boxes.unsqueeze(0), priors_boxes.unsqueeze(1)) # 모든 prior box들과 gt box 하나의 iou 계산해
    '''
    ious:
    [[(gtbox0,prior0 iou), (gtbox1, prior0 iou), (gtbox2, prior0 iou), ... ]
    [(gtbox0,prior1 iou), (gtbox1, prior1 iou), (gtbox2, prior1 iou), ... ]...]
    '''

    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)
    #  to make sure every target has a prior assigned


#TODO : Convert fixed value to cfg value
class DefaultBox():
    def __init__(self, cfg):
        self.img_size = 300 #cfg['img_size']
        self.feature_maps = [38, 19, 10, 5, 3, 1] #cfg['feature_maps']
        self.ar_steps = [4, 6, 6, 6, 4, 4] #cfg['ar_steps']
        self.aspect_ratios = [1, 2, 0.5, 3, 1/3] #cfg['aspect_ratios']
        self.m = len(self.feature_maps)
        self.sk_min = 0.2 #cfg['sk_min']
        self.sk_max = 0.9 #cfg['sk_max']

        if self.sk_min <= 0 or self.sk_max <= self.sk_min:
            raise ValueError('sk_min, sk_max must be grater than 0')
        # from k=1 to k=m+1=7
        self.sk = [self.compute_sk(i) for i in range(1, self.m + 2)]

    def forward(self):
        offset = []
        for k, f_k in enumerate(self.feature_maps):
            for i, j in product(range(f_k), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                cnt = 0
                for ar in self.aspect_ratios:
                    cnt += 1
                    if cnt < self.ar_steps[k]:
                        if ar == 1:
                            w = h = sqrt(self.sk[k] * self.sk[k+1])
                            offset.append([cx, cy, w, h])
                        w = self.sk[k] * sqrt(ar)
                        h = self.sk[k] / sqrt(ar)
                        offset.append([cx, cy, w, h])
        return torch.Tensor(offset) #[num_db, 4]

    def compute_sk(self, k):
        if k == 1:
            return self.sk_min
        elif k == self.m:
            return self.sk_max
        else:
            return round(self.sk_min + (self.sk_max - self.sk_min) / (self.m - 1) * (k - 1), 2)

def hard_negative_mining(loss, gt_labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level
    It keeps all positive predictions and cut the number of negative predictions
    This can lead to faster optimization and a more stable training.

    :param loss: the loss(log softmax) for each example.
    :param gt_labels: Ground Truth labels (N, 8732) = (N, num_priors)
    :param neg_pos_ratio: the ratio between the negative examples and positive examples
    :return: index information of samples that will be used (in gt_labels)
    """
    # loss에는 BackGround일 확률(log softmax값)이 들어간다
    pos_mask = gt_labels > 0 # label0= background, BG가 아닌 곳 mask
    num_pos = pos_mask.long().sum(dim=1, keepdim=True) # (N, 1) 이미지별 positive개수
    num_neg = num_pos * neg_pos_ratio # (N, 1) 이미지별 negative 개수
    loss[pos_mask] = -math.inf  # positive들을 sort에서 밑으로 내려주기 위해
    _, indexes = loss.sort(dim=1, descending=True) # loss 내림차순 정리 -> indices는 sort된 텐서의 원래 인덱스들을 담고있음
    _, orders = indexes.sort(dim=1) # indices를 오름차순으로 다시 sort하면 orders는 원래 loss 텐서의 내림차순에서의 순서가 들어가게 된다.
    neg_mask = orders < num_neg # 내림차순에서의 순서가 negative개수 한계보다 작은 위치 True
    
    return torch.logical_or(neg_mask, pos_mask)


