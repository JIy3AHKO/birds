import torch
from sklearn.metrics import label_ranking_average_precision_score


def lsep_loss_stable(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep


def lsep_loss(input, target, average=True):

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))

    if average:
        return lsep.mean()
    else:
        return lsep

def build_framewise(y_pred, y_true):
    framewise = torch.zeros_like(y_pred['framewise_output'], requires_grad=False)
    framewise_mask = torch.zeros_like(y_pred['framewise_output'], requires_grad=False)

    for sample_id, sample in enumerate(y_true['framewise_target']):
        for s, e, i, is_neg in sample:
            s = int(s * framewise.shape[2])
            e = int(e * framewise.shape[2])
            framewise_mask[sample_id, int(i), s:e] = 1.0
            if not is_neg:
                framewise[sample_id, int(i), s:e] = 1.0

    return framewise, framewise_mask


def iou_continuous(y_pred, y_true, axes=(-1, -2)):
    _EPSILON = 1e-6

    def op_sum(x):
        return x.sum(axes)

    numerator = (op_sum(y_true * y_pred) + _EPSILON)
    denominator = (op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON)
    return numerator / denominator


def train_loss(y_pred, y_true):
    framewise, framewise_mask = build_framewise(y_pred, y_true)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred['framewise_output'],
        framewise,
        reduction='none')

    bce = bce * framewise_mask

    iou = 1 - iou_continuous(torch.sigmoid(y_pred['framewise_output']) * framewise_mask, framewise, axes=-1).mean()
    # iou = iou[framewise.sum(2) > 0].mean()

    # pt = torch.exp(-bce)
    # F_loss = 1.0 * (1 - pt) ** 2 * bce
    # F_loss = F_loss.mean()

    lsep = lsep_loss_stable(y_pred['clipwise_output'], y_true['clipwise_target'])
    framewise_lsep = lsep_loss_stable(y_pred['framewise_output'], framewise)
    loss = lsep * 0.1 + framewise_lsep * 0.01 + bce.mean() * 10 # + iou * 10

    return loss, {'bce': bce.mean().detach().cpu(),
                  'lsep': lsep.mean().detach().cpu(),
                  'iou': iou.mean().detach().cpu(),
                  'framewise_lsep': framewise_lsep.mean().detach().cpu()}

def val_loss(y_pred, y_true):
    lrap = label_ranking_average_precision_score(y_true['clipwise_target'].detach().cpu().numpy(),
                                                 y_pred['clipwise_output'].detach().cpu().numpy())

    return lrap, {'lrap': lrap}