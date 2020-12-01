import torch
import numpy as np
from torch.autograd import Variable, Function


def LWLRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class



def is_intersect(a, b, a1, b1):
    return not ((a < a1 and b < a1) or (a > b1 and b > b1))


class FreeSegmentSet:
    def __init__(self, start_range=(0, 60)):
        self.segments = [start_range]

    def add_segment(self, start, end):
        new_segments = []

        for s in self.segments:
            if is_intersect(start, end, s[0], s[1]):
                if s[0] < start < s[1]:
                    new_segments.append((s[0], start))

                    if end < s[1]:
                        new_segments.append((end, s[1]))

                elif s[0] < end < s[1]:
                    new_segments.append((end, s[1]))

                    if start > s[0]:
                        new_segments.append((s[0], start))

            else:
                new_segments.append(s)

        self.segments = new_segments


def saliency(sample, model):

    model.eval()

    sample['x'].requires_grad_()

    scores = model(sample)['y']

    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]

    score_max.backward()

    saliency, _ = torch.max(sample['x'].grad.data.abs(), dim=1)

    return saliency[0]



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


def spec_augment(spec: torch.Tensor, num_mask=2,
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    for i in range(num_mask):
        _, _, all_frames_num, all_freqs_num = spec.shape
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, :, :, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, :, t0:t0 + num_frames_to_mask, :] = 0

    return spec
