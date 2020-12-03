import argparse
import os
import re

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import torch
import tqdm
from torch.utils.data import DataLoader
import numpy as np

from misc import LWLRAP, lwlrap
from predict import InferenceDataset, pred_extractor_map
from dataset import get_datasets




def get_scores(args):
    model = torch.load(args.model)
    model.cuda()
    model.eval()

    _, val_ds = get_datasets(fold=args.fold, pos_rate=args.pos_rate, duration=args.duration)

    ds = InferenceDataset(val_ds.df[val_ds.df['negative'] == 0], '/datasets/data/birds/train/',
                          duration=args.duration, step=args.step)
    dl = DataLoader(ds, collate_fn=lambda x: x, shuffle=False, batch_size=1, num_workers=12)

    gts = []
    preds = []

    with torch.no_grad():
        for batch_orig in tqdm.tqdm(dl):
            batch = {'x': torch.from_numpy(batch_orig[0]['batch']).cuda()}

            res = model(batch)
            res = pred_extractor_map[args.pred_type](res)
            preds.append(res)
            gts.append(batch_orig[0]['clipwise_target'])

        score_class, weight = lwlrap(np.array(gts), np.array(preds))
        score = (score_class * weight).sum()

        return score, score_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', '-f', type=int, default=0)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--pos_rate', type=float, default=0.75)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--pred_type', type=str, default='clipwise_max')

    args = parser.parse_args()

    if args.fold == -1:
        scores = []
        for fold in range(5):
            args.fold = fold
            foldname_ = re.findall('_fold-\d+_', args.model)[0]
            args.model = args.model.replace(foldname_, f'_fold-{fold}_')
            score, score_class = get_scores(args)
            scores.append(score)
            print(score, score_class)
        print('FOLDS AVG:', np.mean(scores))
    else:
        score, score_class = get_scores(args)
        print(score, score_class)


