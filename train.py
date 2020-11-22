import os

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch
import numpy as np

from opt import AdaBelief, TrapezoidScheduler
from dataset import get_datasets
from trainer import Trainer
from model import Resnet, Effnet
from sklearn.metrics import label_ranking_average_precision_score


def vis_fn(batch, pred):

    imgs = batch['x'][:8].detach().cpu().numpy()

    imgs = imgs * 255
    imgs = imgs.astype('uint8')
    #imgs = imgs.transpose(0, 2, 3, 1)
    print(imgs.shape)

    text = 'GT: '
    for x in batch['y'][:4]:
        xs = x.detach().cpu().numpy()
        st = ", ".join([str(a) for a in xs])
        text += f'[{st}] '

    for x in pred['y'][:4]:
        xs = torch.sigmoid(x).detach().cpu().numpy()
        st = ", ".join([str(a) for a in xs])
        text += f'[{st}] '

    return imgs, text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, default='default')
    parser.add_argument('--fold', '-f', type=int, default=0)

    args = parser.parse_args()
    experiment_name = f'{args.name}_{args.fold}'

    os.mkdir(f'experiments/{experiment_name}/')

    train_size = 1000
    batch_size = 16

    train_ds, val_ds = get_datasets(fold=args.fold)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              num_workers=12)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=12)

    n_epochs = 100

    model = Resnet().cuda()

    def bce_loss(y_pred, y_true):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred['y'],
            y_true['y'],
            reduction='none')

        # pt = torch.exp(-bce)
        # F_loss = 1.0 * (1 - pt) ** 2 * bce
        # F_loss = F_loss.mean()

        return bce.mean(), {'bce': bce.mean()}

    def val_loss(y_pred, y_true):
        bce = f.binary_cross_entropy_with_logits(y_pred['y'], y_true['y'])
        lrap = label_ranking_average_precision_score(y_true['y'].detach().cpu().numpy(),
                                                     torch.sigmoid(y_pred['y']).detach().cpu().numpy())

        return lrap, {'bce': bce, 'lrap': lrap}

    trainer = Trainer(n_epochs,
                      model,
                      bce_loss,
                      val_loss=val_loss,
                      name=args.name,
                      save_fn=lambda x: torch.save(x, f'experiments/{experiment_name}/model.pth'),
                      img_fn=vis_fn,
                      fn_type='image')


    opt = AdaBelief(trainer.model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True, factor=0.66)
    #sch = TrapezoidScheduler(opt, 50).scheduler

    trainer.fit(train_loader, val_loader, opt, sch)
    torch.save(trainer.model, f'models/last_{args.name}.pth')
