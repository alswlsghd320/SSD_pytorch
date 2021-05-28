import torch
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

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


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

if __name__ == '__main__':
    t = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    p = torch.Tensor([[0, 0, 0, 0], [100, 100, 100, 100], [0, 0, 0, 0], [100, 100, 100, 100]])
    # print(t.unsqueeze(0), p.unsqueeze(1))
    # print(torch.min(t.unsqueeze(0)[..., 2:], p.unsqueeze(1)[..., :2]))
    print(iou(t.unsqueeze(0), p.unsqueeze(1)))
