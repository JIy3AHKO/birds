import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import soundfile as sf
import audiomentations as am
from audiomentations.core.transforms_interface import BaseWaveformTransform
from scipy import signal
import librosa as lb

from sklearn.model_selection import KFold
from pysndfx import AudioEffectsChain

from misc import is_intersect, FreeSegmentSet
import audio


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


def preprocess_audio(audio_samples, sample_rate, normalize=False):
    if normalize:
        audio_samples = am.Normalize(p=1.0)(samples=audio_samples, sample_rate=sample_rate)
    sxx = lb.feature.melspectrogram(audio_samples, sr=sample_rate, n_mels=128, hop_length=1024)

    data = lb.power_to_db(sxx)
    return data


def get_target(num_classes, samples, start, end):
    target = np.zeros(num_classes, dtype='float32')

    for i, item in samples.iterrows():
        if item['negative'] == 0 and is_intersect(start, end, item['t_min'], item['t_max']):
            target[item['species_id']] = 1.0

    return target


class PosDataset(Dataset):
    def __init__(self, df, ds_dir='/datasets/data/birds/train/'):
        self.df = df
        self.path = ds_dir
        self.num_classes = 24
        self.sample_rate = 48000
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        for i, (_, item) in enumerate(self.df.iterrows()):
            self.idxs[item['recording_id']].append(i)

        self.overlap = 0.5

    def __getitem__(self, idx):
        idxs = self.idxs[self.ids[idx]]
        samples = self.df.iloc[idxs]

        audio_path = os.path.join(self.path, f'{samples.iloc[0].recording_id}.flac')

        pos_interval_lengths = []
        for label_id in idxs:
            sample = self.df.iloc[label_id]
            pos_interval_lengths.append(sample['t_max'] - sample['t_min'])

        probs = np.array(pos_interval_lengths)
        probs /= probs.sum()

        interval = np.random.choice(np.arange(len(idxs)), p=probs)

        sample = self.df.iloc[idxs[interval]]

        start = np.clip(sample['t_min'] - self.overlap, 0, 60)
        end = np.clip(sample['t_max'] + self.overlap, 0, 60)

        target = get_target(self.num_classes, samples, start, end)

        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)

        audio_sample, sample_rate = sf.read(audio_path,
                                            start=start,
                                            stop=end,
                                            dtype='float32')

        return {'a': audio_sample, 't': target, 'f_min': sample['f_min'], 'f_max': sample['f_max']}

    def __len__(self):
        return len(self.ids)


class NegDataset(Dataset):
    def __init__(self, df, ds_dir='/datasets/data/birds/train/', duration=6.0):
        self.df = df
        self.path = ds_dir
        self.num_classes = 24
        self.sample_rate = 48000
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        for i, (_, item) in enumerate(self.df.iterrows()):
            self.idxs[item['recording_id']].append(i)

        self.overlap = 0.5
        self.duration = duration

    def __getitem__(self, idx):
        idxs = self.idxs[self.ids[idx]]
        samples = self.df.iloc[idxs]

        audio_path = os.path.join(self.path, f'{samples.iloc[0].recording_id}.flac')

        segment_set = FreeSegmentSet()

        for label_id in idxs:
            sample = self.df.iloc[label_id]
            segment_set.add_segment(sample['t_min'] - self.overlap, sample['t_max'] + self.overlap)

        negative_segments = [s for s in segment_set.segments if s[1] - s[0] > self.duration]
        probs = np.array([s[1] - s[0] for s in negative_segments])
        probs /= probs.sum()

        interval = np.random.choice(np.arange(len(negative_segments)), p=probs)

        start, end = negative_segments[interval]

        start = np.random.uniform(start, end - self.duration)
        end = start + self.duration

        target = get_target(self.num_classes, samples, start, end)

        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)

        audio, sample_rate = sf.read(audio_path,
                                     start=start,
                                     stop=end,
                                     dtype='float32')
        if len(audio) == 0:
            raise ValueError()
        return {'a': audio, 't': target}

    def __len__(self):
        return len(self.ids)


class TrainBirdDataset(Dataset):
    def __init__(self, pos_dataset, neg_dataset, duration=6.0, size=5000, normalise=False, sr=48000):
        self.pos_ds = pos_dataset
        self.neg_ds = neg_dataset
        self.duration = duration
        self.size = size
        self.additional_pos_prob = 0.5
        self.additional_pos_count = 3
        self.sr = sr
        self.min_sample_len = 2 * self.sr
        self.sample_crop_limit = 0.5
        self.num_classes = 24
        self.transforms = am.Compose([
            am.AddGaussianSNR(),
            # am.PitchShift(min_semitones=-1, max_semitones=1, p=0.15),
            am.TimeStretch(p=0.15),
            # am.TimeMask(p=0.2, max_band_part=0.33)

        ])
        self.normalise = normalise

    def __getitem__(self, idx):
        neg_id = np.random.randint(0, len(self.pos_ds))

        if np.random.rand() < 0.5:
            noise = self.neg_ds[neg_id]

            audio_sample = noise['a']
            audio_sample = am.Normalize(p=1.0)(samples=audio_sample, sample_rate=self.sr)

            t = noise['t']
        else:
            audio_sample = np.zeros(int(self.sr * self.duration), dtype='float32')
            t = np.zeros(self.num_classes, dtype='float32')

        for i in range(self.additional_pos_count):
            if np.random.rand() > self.additional_pos_prob and i > 0:
                continue
            pos_id = np.random.randint(0, len(self.pos_ds))
            pos = self.pos_ds[pos_id]

            pos['a'] = am.TimeStretch(leave_length_unchanged=False, p=0.15)(samples=pos['a'], sample_rate=self.sr)

            # pos['a'] = audio.audio_filter(pos['a'],
            #                               mode='bandpass',
            #                               w=[pos['f_min'],
            #                                  pos['f_max']],
            #                               wet=np.random.uniform(0.75, 1.0))

            if len(pos['a']) > int(self.duration * self.sr):
                pos['a'] = pos['a'][:int(self.duration * self.sr)]

            position = np.random.uniform(0, (len(audio_sample) - len(pos['a'])) / self.sr)
            alpha = np.random.uniform(0.6, 1.0)
            fade_size = np.random.uniform(0.5, 0.15) * len(pos['a']) / self.sr

            positive_sample = am.Normalize(p=1.0)(samples=pos['a'], sample_rate=self.sr)

            audio_sample = audio.insert(audio_sample, positive_sample, position, alpha=alpha, fade=fade_size)
            t += pos['t']

        audio_sample = self.transforms(samples=audio_sample, sample_rate=self.sr)

        data = preprocess_audio(audio_sample, self.sr, normalize=self.normalise)
        t = np.clip(t, 0, 1)
        return {
            'x': data[None, :],
            'y': t,
        }

    def __len__(self):
        return self.size


class BirdDataset(Dataset):
    def __init__(self, df, ds_dir='/datasets/data/birds/train/', pos_rate=0.5, duration=6.0,
                 disable_negative=False, is_val=False, normalize=False):
        self.df = df
        self.path = ds_dir
        self.transforms = am.Compose([
            am.AddGaussianSNR(),
            am.TimeStretch(p=0.15),
            # am.PitchShift(min_semitones=-1, max_semitones=1, p=0.15),
            # am.TimeMask(p=0.2, max_band_part=0.33)
        ])

        self.num_classes = 24
        self.sample_rate = 48000
        self.duration = duration
        self.pos_rate = pos_rate
        self.epsilon = 0.1
        self.ids = self.df['recording_id'].unique()
        self.idxs = {i: [] for i in self.ids}
        self.is_val = is_val
        self.normalize = normalize

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

        target = get_target(self.num_classes, samples, start, end)

        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        audio_sample, sample_rate = sf.read(audio_path,
                                     start=start,
                                     stop=end,
                                     dtype='float32')

        if not self.is_val:
            # if np.random.rand() < 0.1:
            #     lfreq = np.random.uniform(90, samples.iloc[0]['f_min'])
            #     wet = np.random.uniform(0.5, 1.0)
            #     audio_sample = audio.audio_filter(audio_sample, 'highpass', lfreq, wet)
            #
            # if np.random.rand() < 0.1:
            #     hfreq = np.random.uniform(samples.iloc[0]['f_max'], 20000)
            #     wet = np.random.uniform(0.5, 1.0)
            #     audio_sample = audio.audio_filter(audio_sample, 'lowpass', hfreq, wet)

            audio_sample = self.transforms(samples=audio_sample, sample_rate=sample_rate)
        data = preprocess_audio(audio_sample, self.sample_rate, normalize=self.normalize)

        item = {
            'x': data[None, :],
            'y': target,
        }

        return item

    def __len__(self):
        return len(self.ids)


def get_datasets(seed=1337228, fold=0, n_folds=5, normalize=False, pos_rate=0.75, duration=6.0):
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

    pos_ds = PosDataset(csv_pos[csv_pos['recording_id'].isin(train_ids)])
    neg_ds = NegDataset(csv_pos[csv_pos['recording_id'].isin(train_ids)], duration=duration)

    train_datasets = ConcatDataset([
        BirdDataset(train_df, pos_rate=pos_rate, disable_negative=False, normalize=normalize, duration=duration),
        TrainBirdDataset(pos_ds, neg_ds, normalise=normalize, duration=duration)
    ])

    return train_datasets, BirdDataset(test_df, pos_rate=1.0, is_val=True, normalize=normalize, duration=duration)
