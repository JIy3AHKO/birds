import argparse
import os

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import torch
import tqdm
from torch.utils.data import DataLoader
import numpy as np

from misc import LWLRAP, lwlrap
from predict import InferenceDataset
from dataset import get_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, default='default')
    parser.add_argument('--fold', '-f', type=int, default=0)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--pos_rate', type=float, default=0.75)

    args = parser.parse_args()
    args.normalize = bool(args.normalize)

    model = torch.load(args.model)
    model.cuda()
    model.eval()

    _, val_ds = get_datasets(fold=args.fold, normalize=args.normalize, pos_rate=args.pos_rate)

    ds = InferenceDataset(val_ds.df[val_ds.df['negative'] == 0], '/datasets/data/birds/train/')
    dl = DataLoader(ds, collate_fn=lambda x: x, shuffle=False, batch_size=1, num_workers=12)

    gts = []
    preds = []

    with torch.no_grad():
        for batch_orig in tqdm.tqdm(dl):
            batch = {'x': torch.from_numpy(batch_orig[0]['batch']).cuda()}

            res = model(batch)
            res = torch.sigmoid(res['y'])
            res = res.detach().cpu().numpy()
            res = res.max(0)

            preds.append(res)
            gts.append(batch_orig[0]['target'])

        score_class, weight = lwlrap(np.array(gts), np.array(preds))
        score = (score_class * weight).sum()
        print(score, score_class)

