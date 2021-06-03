from numpy import sqrt
import torch
from itertools import product as product
import math
from configs import ssd300 as cfg

def cxcy_to_xyxy(cxcy):
    """
    Convert bbox from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    :param cxcy:bbox coordinates having center-size coordinates, a tensor of size(n_boxes, 4)
    :return: bbox in (xmin, ymin, xmax, ymax) coordinates
    """
    return torch.cat([cxcy[..., :2] - (cxcy[..., 2:] / 2), cxcy[..., :2] + (cxcy[..., 2:] / 2)], -1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy (num_defaults, 4): bounding boxes in center-size coordinates
    :param priors_cxcy (num_defaults, 4): prior boxes with respect to which the encoding must be performed
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:]),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:])], 1)  # g_w, g_h

def intersect(box_1, box_2):
    '''
    :param box_1: [num_objects, 4(xmin, ymin, xmax, ymax)]
    :param box_2: [num_defaults, 4]
    :return: [num_objects, num_defaults]
    '''
    A = box_1.size(0) #num_objects
    B = box_2.size(0) #num_defaults
    min_xy = torch.max(box_1[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_2[:, :2].unsqueeze(0).expand(A, B, 2)) # [A, B, 2(x_min, y_min)]

    max_xy = torch.min(box_1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_2[:, 2:].unsqueeze(0).expand(A, B, 2)) # [A, B, 2(x_max, y_max)]

    inter = torch.clamp((max_xy - min_xy), min=0)  # [A, B, 2(x_max - x_min, y_max - y_min)]

    return inter[:, :, 0] * inter[:, :, 1] # [A, B] (x_max - x_min) * (y_max - y_min)

def IOU(box_1, box_2):
    '''
    :param box_1: Ground truth bounding box ; [num_objects, 4(xmin, ymin, xmax, ymax)]
    :param box_2: Prior boxes ; [num_defaults, 4]
    :return: IOU(A, B) = A ∩ B / A ∪ B
                       = A ∩ B / (A + B - A ∩ B) ; [box_1.shape[0], box_2.shape[0]]
    '''
    inter = intersect(box_1, box_2)
    area_a = ((box_1[:, 2] - box_1[:, 0]) *
              (box_1[:, 3] - box_1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_2[:, 2] - box_2[:, 0]) *
              (box_2[:, 3] - box_2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class DefaultBox():
    def __init__(self):
        self.feature_maps = cfg.FEATURE_MAPS
        self.ar_steps = cfg.AR_STEPS
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.m = len(self.feature_maps)
        self.sk_min = cfg.SK_MIN
        self.sk_max = cfg.SK_MAX

        if self.sk_min <= 0 or self.sk_max <= self.sk_min:
            raise ValueError('sk_min, sk_max must be grater than 0')
        # from k=1 to k=m+1=7
        self.sk = [self.compute_sk(i) for i in range(1, self.m + 2)]

    def __call__(self):
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
                            w = h = sqrt(self.sk[k] * self.sk[k + 1])
                            offset.append([cx, cy, w, h])
                        w = self.sk[k] * sqrt(ar)
                        h = self.sk[k] / sqrt(ar)
                        offset.append([cx, cy, w, h])
        return torch.Tensor(offset) #[num_defaults, 4]

    def compute_sk(self, k):
        if k == 1:
            return self.sk_min
        elif k == self.m:
            return self.sk_max
        else:
            return round(self.sk_min + (self.sk_max - self.sk_min) / (self.m - 1) * (k - 1), 2)


def hard_negative_mining(loss, gt_labels, neg_pos_ratio, device):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level
    It keeps all positive predictions and cut the number of negative predictions
    This can lead to faster optimization and a more stable training.
    :param loss: the loss(log softmax) for each example.
    :param gt_labels: Ground Truth labels (N, 8732) = (N, num_defaults)
    :param neg_pos_ratio: the ratio between the negative examples and positive examples
    :return: index information of samples that will be used (in gt_labels)
    """
    # loss에는 BackGround일 확률(log softmax값)이 들어간다
    pos_mask = gt_labels > 0  # label0= background, BG가 아닌 곳 mask
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)  # (N, 1) 이미지별 positive개수
    num_neg = num_pos * neg_pos_ratio  # (N, 1) 이미지별 negative 개수
    loss[pos_mask] = -math.inf  # positive들을 sort에서 밑으로 내려주기 위해
    _, indexes = loss.sort(dim=1, descending=True)  # loss 내림차순 정리 -> indices는 sort된 텐서의 원래 인덱스들을 담고있음
    _, orders = indexes.sort(dim=1)  # indices를 오름차순으로 다시 sort하면 orders는 원래 loss 텐서의 내림차순에서의 순서가 들어가게 된다.
    neg_mask = orders < num_neg  # 내림차순에서의 순서가 negative개수 한계보다 작은 위치 True

    return torch.logical_or(neg_mask, pos_mask)