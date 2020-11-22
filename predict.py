import argparse
import os

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import tqdm

from dataset import preprocess_audio
from model import Resnet
import soundfile as sf

duration = 6
nperseg = 1032
sample_rate = 48000


class InferenceDataset(Dataset):
    def __init__(self, sub, dir):
        self.ids = sub['recording_id'].values
        self.dir = dir

    def __getitem__(self, item):
        audio, sample_rate = sf.read(os.path.join(self.dir, f"{self.ids[item]}.flac"), dtype='float32')

        batch = []

        for start in np.arange(0, 60, duration):
            start = np.clip(start, 0, 60 - duration)
            a = audio[int(start * sample_rate):int((start + duration) * sample_rate)]
            data = preprocess_audio(a, nperseg, sample_rate)
            batch.append(data[None, None, :])

        batch = np.concatenate(batch, axis=0)

        return {'batch': batch, 'recording_id': self.ids[item]}

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--dir', '-d', type=str, default='/datasets/data/birds/test/')

    args = parser.parse_args()

    model = torch.load(args.model)
    model.cuda()
    model.eval()

    rows = []

    sub = pd.read_csv('/datasets/data/birds/sample_submission.csv')
    ds = InferenceDataset(sub, args.dir)

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

