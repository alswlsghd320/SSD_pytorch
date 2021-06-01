import torch
from utils.default_box_utils import cxcy_to_xyxy, IOU

def match(threshold, truths, priors, labels, loc_true, conf_true, idx):
    """

    :param threshold: the value when labeling background
    :param truths: ground truth boxes [num_obj, num_defaults]
    :param priors: Defaults boxes [num_defaults, 4]
    :param labels: [num_obj]
    :param loc_true: [num_defaults, 4] encoded offsets
    :param conf_true: [num_defaults]top class label for each prior
    :param idx: current index of the batch
    """

    # calculate IOU
    overlaps = IOU(truths, cxcy_to_xyxy(priors)) #[num_obj, num_defaults]

    # [1, num_obj] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
    # [1, num_defaults] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
    # ground truth에 대해 가장 많이 겹치는 defaults box 인덱스 부분을 2(1보다 큰 다른 값이면 됨)로 채움
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # [num_defaults,4]
    conf = labels[best_truth_idx] + 1         # [num_defaults]
    conf[best_truth_overlap < threshold] = 0  # background
    g_loc = encode(matches, priors)
    loc_true[idx] = g_loc    # [num_defaults, 4] encoded offsets to learn
    conf_true[idx] = conf  # [num_defaults] top class label for each prior

def encode(matched, priors):
    """
    :param matched: [num_defaults, 4(x_min, y_min, x_max, y_max)].
    :param priors: [num_defaults,4]
    :return: [num_defaults, 4]
    """

    # match center - prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh)
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_defaults, 4(cx, cy, w, h)]

def decode(loc, priors):
    """
    :param loc: [num_defaults, 4]
    :param priors: [num_defaults, 4]
    :return: [num_defaults, 4(x_min, y_min, x_max, y_max)]
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes

def get_matching_label(loc_t, conf_t, priors, threshold=0.5):
    num_obj = len(loc_t)
    num_defaults = priors.size(0)

    loc_true = torch.Tensor(num_obj, num_defaults, 4)
    conf_true = torch.LongTensor(num_obj, num_defaults)
    for i in range(num_obj):
        locs = loc_t[i]
        labels = conf_t[i]
        defaults = priors.data
        match(threshold, locs, defaults, labels, loc_true, conf_true, i)

    return loc_true, conf_true

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count