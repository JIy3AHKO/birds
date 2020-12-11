import os
from functools import partial

from torch.utils.data.dataloader import default_collate

from loss import train_loss, val_loss

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

from opt import AdaBelief, TrapezoidScheduler
from dataset import get_datasets
from trainer import Trainer
from model import Resnet, Effnet, get_model


def generate_bgr_palette(count: int, scale: float = 1 / 255, as_hex=False) -> list:
    """
    Return different colors in bgr colorspace :param count: number colors
    :param scale: color scale
    :return list of colors
    """
    ret = []
    for i in range(count):
        hsv_color = np.array([i / count * 180, 255, 255]).astype(np.uint8)
        bgr_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
        if as_hex:
            rgb = sum(255 ** b * c for b, c in enumerate(bgr_color))
            ret.append('%0.6X' % rgb)
        else:
            ret.append(bgr_color)
    return ret


PALETTE = generate_bgr_palette(24, scale=1.0)


def collate_fn(batch):
    framewise_targets = [s.pop('framewise_target') for s in batch]

    batch = default_collate(batch)
    batch['framewise_target'] = framewise_targets

    return batch


def draw_bars_gt(h, w, framewise_target):
    bar = np.zeros((1, 3, h * 24, w), dtype='uint8')

    for s, e, i, is_neg in framewise_target:
        s = int(w * s)
        e = int(w * e)

        bar[0, :, int(h * i):int(h * (i + 1)), s:e] = PALETTE[int(i)][:, None, None]
        if is_neg:
            bar[0, :, int(h * i):int(h * (i + 1)), s:e:2] //= 2

    return bar


def draw_bars_pred(h, framewise_output):
    probs = (torch.sigmoid(framewise_output)).detach().cpu().repeat_interleave(dim=1, repeats=h).float().numpy()

    probs = probs[:, None] * np.repeat(np.array(PALETTE).T, axis=1, repeats=h)[None, :, :, None]

    return probs.astype('uint8')


def vis_fn(batch, pred):

    imgs = pred['spec'][:4].repeat(1, 3, 1, 1).detach().cpu().numpy()

    bars = []
    for x in batch['framewise_target'][:4]:
        bars.append(draw_bars_gt(10, imgs.shape[3], x))

    bars_gt = np.concatenate(bars)
    bars_pred = draw_bars_pred(10, pred['framewise_output'][:4])

    imgs -= np.min(imgs, axis=(1, 2, 3), keepdims=True)
    imgs /= np.max(imgs, axis=(1, 2, 3), keepdims=True)
    maxcol = np.array([255, 255, 255])[None, :, None, None]
    mincol = np.array([0, 0, 0])[None, :, None, None]
    imgs = imgs * maxcol + (1 - imgs) * mincol

    imgs = imgs.astype('uint8')
    imgs = np.concatenate([imgs, bars_gt, bars_pred], axis=-2)

    text = 'GT: '
    for x in batch['clipwise_target'][:4]:
        xs = x.detach().cpu().numpy()
        st = ", ".join([str(a) for a in xs])
        text += f'[{st}] '

    for x in pred['clipwise_output'][:4]:
        xs = torch.sigmoid(x).detach().cpu().numpy()
        st = ", ".join([str(a) for a in xs])
        text += f'[{st}] '

    return imgs, text, batch['x'][:4]


def parse_aug_args(arguments):
    return {
        'gaussian_snr': {
            'max_SNR': arguments.gaussian_snr_max,
            'p': arguments.gaussian_snr_p
        },
        'gain': {
            'min_gain_in_db': -arguments.random_gain_r,
            'max_gain_in_db': arguments.random_gain_r,
            'p': arguments.random_gain_p
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, default='default')
    parser.add_argument('--fold', '-f', type=int, default=0)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str, default="resnet34")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--pos_rate', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--dssize', type=int, default=10000)
    parser.add_argument('--gaussian_snr_max', type=float, default=0.4)
    parser.add_argument('--gaussian_snr_p', type=float, default=0.4)
    parser.add_argument('--specaug_freq_drop', type=float, default=0.0)
    parser.add_argument('--specaug_time_drop', type=float, default=0.0)
    parser.add_argument('--random_gain_r', type=float, default=10.0)
    parser.add_argument('--random_gain_p', type=float, default=1.0)
    parser.add_argument('--shift_gain', type=float, default=-15.0)

    args = parser.parse_args()
    experiment_name = ""

    for k, v in vars(args).items():
        k_short = ''.join([x[0] for x in k.split('_')])
        experiment_name += f"{k_short}-{v}_" if k != "name" else f"{v}_"

    if args.debug and os.path.exists(f'experiments/{experiment_name}/'):
        os.removedirs(f'experiments/{experiment_name}/')
    os.mkdir(f'experiments/{experiment_name}/')

    train_size = 1000
    batch_size = args.bs

    train_ds, val_ds = get_datasets(fold=args.fold,
                                    pos_rate=args.pos_rate,
                                    duration=args.duration,
                                    dssize=args.dssize,
                                    aug_params=parse_aug_args(args))
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn,
                              num_workers=12 if not args.debug else 0)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            drop_last=True,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            num_workers=12 if not args.debug else 0)

    n_epochs = 40

    model = get_model(
        name=args.model,
        dropout=args.dropout,
        time_drop=args.specaug_time_drop,
        freq_drop=args.specaug_freq_drop
    ).cuda()

    trainer = Trainer(n_epochs,
                      model,
                      train_loss,
                      val_loss=val_loss,
                      name=experiment_name,
                      save_fn=lambda x: torch.save(x, f'experiments/{experiment_name}/model.pth'),
                      img_fn=vis_fn,
                      fn_type='image')

    opt = AdaBelief(trainer.model.parameters(), lr=args.lr, weight_decay=args.wd, weight_decouple=True)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, verbose=True, factor=0.33)
    #sch = TrapezoidScheduler(opt, 50).scheduler

    trainer.fit(train_loader, val_loader, opt, sch)
