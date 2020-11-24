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
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str, default="resnet34")
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--pos_rate', type=float, default=0.75)

    args = parser.parse_args()
    experiment_name = ""
    for k, v in vars(args).items():
        experiment_name += f"{k}-{v}_" if k != "name" else f"{v}_"

    if args.debug and os.path.exists(f'experiments/{experiment_name}/'):
        os.removedirs(f'experiments/{experiment_name}/')
    os.mkdir(f'experiments/{experiment_name}/')

    train_size = 1000
    batch_size = args.bs

    train_ds, val_ds = get_datasets(fold=args.fold, )
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              num_workers=12 if not args.debug else 0)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=12 if not args.debug else 0)

    n_epochs = 40

    model = Resnet(model_type=args.model, dropout=args.dropout).cuda()

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
                      name=experiment_name,
                      save_fn=lambda x: torch.save(x, f'experiments/{experiment_name}/model.pth'),
                      img_fn=vis_fn,
                      fn_type='image')

    opt = AdaBelief(trainer.model.parameters(), lr=args.lr, weight_decay=args.wd, weight_decouple=True)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True, factor=0.66)
    #sch = TrapezoidScheduler(opt, 50).scheduler

    trainer.fit(train_loader, val_loader, opt, sch)
    torch.save(trainer.model, f'models/last_{args.name}.pth')
