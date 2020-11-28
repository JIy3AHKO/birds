import argparse
import os
import re

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import tqdm

from dataset import preprocess_audio, get_target
from model import Resnet
import soundfile as sf

sample_rate = 48000


class Ensamble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        res = [m(x) for m in self.models]

        return {'y': torch.mean(torch.cat([r['y'][None] for r in res], dim=0), dim=0)}


class InferenceDataset(Dataset):
    def __init__(self, sub, dir, duration, normalize):
        self.dir = dir
        self.df = sub
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        for i, (_, item) in enumerate(self.df.iterrows()):
            self.idxs[item['recording_id']].append(i)

        self.num_classes = 24
        self.duration = duration
        self.normalize = normalize

    def __getitem__(self, idx):
        idxs = self.idxs[self.ids[idx]]
        samples = self.df.iloc[idxs]

        audio, sample_rate = sf.read(os.path.join(self.dir, f"{samples.iloc[0]['recording_id']}.flac"), dtype='float32')
        item = {}
        if 'species_id' in samples.iloc[0]:
            item['target'] = get_target(self.num_classes, samples, 0, 60)

        batch = []

        for start in np.arange(0, 60, self.duration):
            start = np.clip(start, 0, 60 - self.duration)
            a = audio[int(start * sample_rate):int((start + self.duration) * sample_rate)]
            data = preprocess_audio(a, sample_rate, normalize=self.normalize)
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
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--allfolds', action='store_true')

    args = parser.parse_args()
    args.normalize = bool(args.normalize)


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
    ds = InferenceDataset(sub, args.dir, duration=args.duration, normalize=args.normalize)

    dl = DataLoader(ds, collate_fn=lambda x: x, shuffle=False, batch_size=1, num_workers=12)
    with torch.no_grad():
        for batch_orig in tqdm.tqdm(dl):
            batch = {'x': torch.from_numpy(batch_orig[0]['batch']).cuda()}

            res = model(batch)
            res = torch.sigmoid(res['y'])
            res = res.detach().cpu().numpy()
            res = res.max(0)

            row = [batch_orig[0]['recording_id'], *res]
            rows.append(row)

    sub = pd.DataFrame(rows, columns=['recording_id'] + [f's{i}' for i in range(24)])
    sub.to_csv('submission.csv', index=False)

