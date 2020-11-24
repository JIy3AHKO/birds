import torch
import numpy as np

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

