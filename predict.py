import argparse
import os

import torch
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import tqdm

from model import Resnet
import soundfile as sf


duration = 6
nperseg = 1032
sample_rate = 48000

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
    with torch.no_grad():
        for i, item in tqdm.tqdm(sub.iterrows(), total=len(sub)):
            audio, sample_rate = sf.read(os.path.join(args.dir, f"{item['recording_id']}.flac"), dtype='float32')

            batch = []

            for start in np.arange(0, 60, duration):
                f, t, sxx = spectrogram(audio[start * sample_rate:(start + duration) * sample_rate],
                                        nperseg=nperseg,
                                        fs=sample_rate)

                freq_idx = np.max(np.argwhere(f <= 14000))

                sxx = sxx[:freq_idx]

                data = -np.log10(sxx + 1e-16)

                data /= data.max()
                data -= data.min()

                batch.append(data[None, None, :])

            batch = np.concatenate(batch, axis=0)

            batch = {'x': torch.from_numpy(batch).cuda()}

            res = model(batch)
            res = torch.sigmoid(res['y'])
            res = res.detach().cpu().numpy()
            res = res.max(0)

            row = [item['recording_id'], *res]
            rows.append(row)

    sub = pd.DataFrame(rows, columns=['recording_id'] + [f's{i}' for i in range(24)])
    sub.to_csv('submission.csv', index=False)

