import argparse
import os
import re

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tqdm

from dataset import preprocess_audio, get_target
import soundfile as sf


def extract_clipwise_max(res):
    res = torch.sigmoid(res['clipwise_output'])
    res = res.detach().cpu().numpy()
    res = res.max(0)

    return res


def extract_clipwise_perc(res, perc=95):
    res = torch.sigmoid(res['clipwise_output'])
    res = res.detach().cpu().numpy()
    res = np.percentile(res, axis=0, q=perc)

    return res


def extract_framewise_max(res):
    res = torch.sigmoid(res['framewise_output'])
    res = res.detach().cpu().numpy()
    res = res.max(-1).max(0)

    return res


def extract_framewise_mean(res):
    res = torch.sigmoid(res['framewise_output'])
    res = res.detach().cpu().numpy()
    res = res.mean(-1).max(0)

    return res


pred_extractor_map = {
    'clipwise_max': extract_clipwise_max,
    'clipwise_perc': extract_clipwise_perc,
    'framewise_max': extract_framewise_max,
    'framewise_mean': extract_framewise_mean,
}


sample_rate = 48000


class Ensamble(torch.nn.Module):
    def __init__(self, models, avg_mode='mean'):
        super().__init__()
        self.models = models
        self.avg_mode = avg_mode

    def forward(self, x):
        res = [m(x) for m in self.models]
        if self.avg_mode == 'geomean':
            return {'clipwise_output': torch.prod(torch.cat([torch.sigmoid(r['clipwise_output'][None]) + 1e-8 for r in res], dim=0), dim=0) ** (1 / len(self.models))}
        elif self.avg_mode == 'mean':
            return {'clipwise_output': torch.sigmoid(torch.mean(torch.cat([r['clipwise_output'][None] for r in res], dim=0), dim=0))}
        else:
            raise ValueError(f"Unsupported {self.avg_mode}")


class InferenceDataset(Dataset):
    def __init__(self, sub, dir, duration, step=None):
        self.dir = dir
        self.df = sub
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        for i, (_, item) in enumerate(self.df.iterrows()):
            self.idxs[item['recording_id']].append(i)

        self.num_classes = 24
        self.duration = duration
        self.step = step or duration

    def __getitem__(self, idx):
        idxs = self.idxs[self.ids[idx]]
        samples = self.df.iloc[idxs]

        audio, sample_rate = sf.read(os.path.join(self.dir, f"{samples.iloc[0]['recording_id']}.flac"), dtype='float32')
        item = {}
        if 'species_id' in samples.iloc[0]:
            item['clipwise_target'], item['framewise_target'] = get_target(self.num_classes, samples, 0, 60)

        batch = []

        for start in np.arange(0, 60, self.step):
            if start + self.duration > 60:
                break
            start = np.clip(start, 0, 60 - self.duration)
            a = audio[int(start * sample_rate):int((start + self.duration) * sample_rate)]
            data = preprocess_audio(a)
            batch.append(data[None, None, :])

        batch = np.concatenate(batch, axis=0)

        item['batch'] = batch
        item['recording_id'] = samples.iloc[0]['recording_id']

        return item

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--dir', '-d', type=str, default='/datasets/data/birds/test/')
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--allfolds', action='store_true')
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--pred_type', type=str, default='clipwise_max')

    args = parser.parse_args()

    if args.allfolds:
        models = []

        for fold in range(5):
            args.fold = fold
            foldname_ = re.findall('_fold-\d+_', args.model)[0]
            args.model = args.model.replace(foldname_, f'_fold-{fold}_')
            model = torch.load(args.model)
            model.cuda()
            model.eval()
            models.append(model)
        model = Ensamble(models)
    else:
        model = torch.load(args.model)
        model.cuda()
        model.eval()

    rows = []

    sub = pd.read_csv('/datasets/data/birds/sample_submission.csv')
    ds = InferenceDataset(sub, args.dir, duration=args.duration, step=args.step)

    dl = DataLoader(ds, collate_fn=lambda x: x, shuffle=False, batch_size=1, num_workers=12)
    with torch.no_grad():
        for batch_orig in tqdm.tqdm(dl):
            batch = {'x': torch.from_numpy(batch_orig[0]['batch']).cuda()}

            res = model(batch)
            res = pred_extractor_map[args.pred_type](res)

            row = [batch_orig[0]['recording_id'], *res]
            rows.append(row)

    sub = pd.DataFrame(rows, columns=['recording_id'] + [f's{i}' for i in range(24)])
    sub.to_csv('submission.csv', index=False)

