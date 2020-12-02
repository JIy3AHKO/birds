import os
from functools import partial

from torch.utils.data.dataloader import default_collate

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


from misc import lsep_loss_stable as lsep_loss

import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch
import numpy as np
import cv2

from opt import AdaBelief, TrapezoidScheduler
from dataset import get_datasets
from trainer import Trainer
from model import Resnet, Effnet, get_model
from sklearn.metrics import label_ranking_average_precision_score


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

    for s, e, i in framewise_target:
        s = int(w * s)
        e = int(w * e)

        bar[0, :, int(h * i):int(h * (i + 1)), s:e] = PALETTE[int(i)][:, None, None]

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

    imgs /= np.max(imgs, axis=0, keepdims=True)
    imgs *= 255

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

    return imgs, text


def parse_aug_args(arguments):
    return {
        'gaussian_snr': {
            'max_SNR': arguments.gaussian_snr_max,
            'p': arguments.gaussian_snr_p
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, default='default')
    parser.add_argument('--fold', '-f', type=int, default=0)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str, default="resnet34")
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--pos_rate', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--dssize', type=int, default=10000)
    parser.add_argument('--gaussian_snr_max', type=float, default=0.4)
    parser.add_argument('--gaussian_snr_p', type=float, default=0.4)

    args = parser.parse_args()
    experiment_name = ""

    for k, v in vars(args).items():
        experiment_name += f"{k}-{v}_" if k != "name" else f"{v}_"

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

    n_epochs = 30

    model = get_model(name=args.model, dropout=args.dropout).cuda()


    def iou_continuous(y_pred, y_true, axes=(-1, -2)):
        _EPSILON = 1e-6

        def op_sum(x):
            return x.sum(axes)

        numerator = (op_sum(y_true * y_pred) + _EPSILON)
        denominator = (op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred) + _EPSILON)
        return numerator / denominator

    def bce_loss(y_pred, y_true):
        framewise = torch.zeros_like(y_pred['framewise_output'], requires_grad=False)

        for sample_id, sample in enumerate(y_true['framewise_target']):
            for s, e, i in sample:
                s = int(s * framewise.shape[2])
                e = int(e * framewise.shape[2])
                framewise[sample_id, int(i), s:e] = 1.0

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred['framewise_output'],
            framewise,
            reduction='none')

        iou = 1 - iou_continuous(torch.sigmoid(y_pred['framewise_output']), framewise, axes=-1)
        iou = iou[framewise.sum(2) > 0].mean()

        # pt = torch.exp(-bce)
        # F_loss = 1.0 * (1 - pt) ** 2 * bce
        # F_loss = F_loss.mean()

        lsep = lsep_loss(y_pred['clipwise_output'], y_true['clipwise_target'])
        loss = lsep + iou + bce.mean()

        return loss, {'bce': bce.mean(), 'lsep': lsep, 'iou': iou}

    def val_loss(y_pred, y_true):
        framewise = torch.zeros_like(y_pred['framewise_output'], requires_grad=False)

        for sample_id, sample in enumerate(y_true['framewise_target']):
            for s, e, i in sample:
                s = int(s * framewise.shape[2])
                e = int(e * framewise.shape[2])
                framewise[sample_id, int(i), s:e] = 1.0

        iou = iou_continuous(torch.sigmoid(y_pred['framewise_output']), framewise, axes=-1)
        iou = iou[framewise.sum(-1) > 0].mean()

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred['framewise_output'],
            framewise)
        lrap = label_ranking_average_precision_score(y_true['clipwise_target'].detach().cpu().numpy(),
                                                     torch.sigmoid(y_pred['clipwise_output']).detach().cpu().numpy())
        lsep = lsep_loss(y_pred['clipwise_output'], y_true['clipwise_target'])

        return lrap, {'bce': bce, 'lrap': lrap, 'lsep': lsep, 'iou': iou}

    trainer = Trainer(n_epochs,
                      model,
                      bce_loss,
                      val_loss=val_loss,
                      name=experiment_name,
                      save_fn=lambda x: torch.save(x, f'experiments/{experiment_name}/model.pth'),
                      img_fn=vis_fn,
                      fn_type='image')

    opt = AdaBelief(trainer.model.parameters(), lr=args.lr, weight_decay=args.wd, weight_decouple=True)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, verbose=True, factor=0.33)
    #sch = TrapezoidScheduler(opt, 50).scheduler

    trainer.fit(train_loader, val_loader, opt, sch)
    torch.save(trainer.model, f'models/last_{args.name}.pth')
