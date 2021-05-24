import torch
import math

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