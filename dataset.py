import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
import audiomentations as am
from audiomentations.core.transforms_interface import BaseWaveformTransform
from scipy.signal import spectrogram

import librosa as lb

from sklearn.model_selection import KFold
from pysndfx import AudioEffectsChain


class SndTransform(BaseWaveformTransform):
    def __init__(self, transform, p=0.5, **kwargs):
        super().__init__(p)
        self.args = kwargs
        self.func = transform

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            for k, rng in self.args:
                self.parameters[f'transform_parameter_{k}'] = np.random.uniform(*rng)

    def apply(self, samples, sample_rate):
        f = self.func(**{k.replace('transform_parameter_', ''): v
                         for k, v
                         in self.parameters.items()
                         if 'transform_parameter_' in k})

        return f(samples)


def is_intersect(a, b, a1, b1):
    return not ((a < a1 and b < a1) or (a > b1 and b > b1))


def preprocess_audio(audio, nperseg, sample_rate):
    sxx = lb.feature.melspectrogram(audio, sr=sample_rate, n_mels=128, hop_length=1024)
    # freq_idx = np.max(np.argwhere(f <= 14000))
    #
    # sxx = sxx[:freq_idx]

    data = np.log10(sxx + 1e-16)
    return data


class BirdDataset(Dataset):
    def __init__(self, df, ds_dir='/datasets/data/birds/train/', pos_rate=0.5, duration=6, nperseg=1032,
                 disable_negative=False, is_val=False):
        self.df = df
        self.path = ds_dir
        self.transforms = am.Compose([
            am.AddGaussianSNR(),
         #   am.Normalize(p=1.0)
        ])

        self.num_classes = 24
        self.sample_rate = 48000
        self.duration = duration
        self.pos_rate = pos_rate
        self.epsilon = 0.1
        self.nperseg = nperseg
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        self.is_val = is_val

        for i, (_, item) in enumerate(self.df.iterrows()):
            self.idxs[item['recording_id']].append(i)

        if disable_negative:
            new_ids = []
            for i in self.ids:
                negatives = self.df.iloc[self.idxs[i]]['negative']
                if (1 - negatives).sum() > 0:
                    new_ids.append(i)

            self.ids = new_ids
            self.idxs = {k: v for k, v in self.idxs.items() if k in self.ids}

    def __getitem__(self, idx):
        idxs = self.idxs[self.ids[idx]]
        samples = self.df.iloc[idxs]

        audio_path = os.path.join(self.path, f'{samples.iloc[0].recording_id}.flac')

        if np.random.rand() < self.pos_rate:
            sample = self.df.iloc[np.random.choice(idxs)]
            end = np.random.uniform(sample['t_min'] + self.epsilon, sample['t_max'] + self.duration - self.epsilon)
            end = np.clip(end, 0, 60)
            start = np.clip(end - self.duration, 0, 60)
            end = start + self.duration
        else:
            # TODO: choose between labeled intervals
            start = np.random.uniform(0, 60 - self.duration - self.epsilon)
            end = start + self.duration

        target = np.zeros(self.num_classes, dtype='float32')

        for i, item in samples.iterrows():
            if item['negative'] == 0 and is_intersect(start, end, item['t_min'], item['t_max']):
                target[item['species_id']] = 1.0

        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        audio, sample_rate = sf.read(audio_path,
                                     start=start,
                                     stop=end,
                                     dtype='float32')

        if not self.is_val:

            audio = self.transforms(samples=audio, sample_rate=sample_rate)
        data = preprocess_audio(audio, self.nperseg, self.sample_rate)

        item = {
            'x': data[None, :],
            'y': target
        }

        return item

    def __len__(self):
        return len(self.ids)


def get_datasets(seed=1337228, fold=0, n_folds=5):
    csv_pos = pd.read_csv('/datasets/data/birds/train_tp_prep.csv')
    csv_neg = pd.read_csv('/datasets/data/birds/train_fp_prep.csv')

    csv_pos['negative'] = 0
    csv_neg['negative'] = 1

    df = pd.concat([csv_pos, csv_neg])

    ids = df['recording_id'].unique()

    kf = KFold(n_folds, shuffle=True, random_state=seed)

    fold_gen = kf.split(ids)

    train_idx, val_idx = list(fold_gen)[fold]

    train_ids, test_ids = ids[train_idx], ids[val_idx]

    train_df = df[df['recording_id'].isin(train_ids)]
    test_df = df[df['recording_id'].isin(test_ids)]

    return BirdDataset(train_df, pos_rate=0.75, disable_negative=False), BirdDataset(test_df, pos_rate=1.0, is_val=True)



