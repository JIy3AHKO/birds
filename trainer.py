from collections import defaultdict

import tqdm
import torch
import numpy as np
import visdom


class VisdomLineOp:
    def __init__(self, name):
        self.xs = []
        self.ys = []
        self.name = name

    def set_data(self, value):
        if (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 2:
            x, y = value

            self.xs = x
            self.ys = y
        else:
            self.xs, self.ys = [], value

    def append(self, value):
        if (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 2:
            x, y = value

            self.xs.append(x)
            self.ys.append(y)
        else:
            self.ys.append(value)

    def draw(self, vis: visdom.Visdom):
        if len(self.xs) == 0:
            xs = np.arange(len(self.ys))
        else:
            xs = self.xs
        vis.line(self.ys, xs, name=self.name, opts={'title': self.name}, win=self.name)


class VisdomImageOp:
    def __init__(self, name, nrows=None, padding=None):
        self.imgs = []
        self.name = name
        self.nrows = nrows
        self.padding = padding

    def set_data(self, value):
        self.imgs = [value]

    def append(self, value):
        self.imgs.append(value)

    def draw(self, vis: visdom.Visdom):
        for res in self.imgs:
            if len(res) == 2:
                img, text = res
                audio = None
            elif len(res) == 3:
                img, text, audio = res
            else:
                raise ValueError(f"Visdom fn should return 2 or 3 items.")
            vis.images(img, nrow=img.shape[0], padding=self.padding, win=self.name)
            vis.text(text, win=self.name + 'text')
            if audio is not None:
                for i, a in enumerate(audio):
                    vis.audio(a[0], win=self.name + f'audio_{i}', opts={
                        'sample_frequency': 48000,
                        'title': f'{self.name}_audio_{i}'})


class VisdomVideoOp:
    def __init__(self, name, nrows=None, padding=None):
        self.videos = []
        self.name = name
        self.nrows = nrows
        self.padding = padding

    def set_data(self, value):
        self.videos = value

    def append(self, value):
        self.videos = np.concatenate([self.videos, value], axis=0)

    def draw(self, vis: visdom.Visdom):
        print('video sent')
        for v in self.videos:
            vis.video(v, win=self.name)


class VisdomLogger:
    ops = {
        'line': VisdomLineOp,
        'image': VisdomImageOp,
        'video': VisdomVideoOp,
    }

    def __init__(self, env):
        self.visdom_logger = visdom.Visdom(env=env)
        self.plots = {}
        self.env = env

    def add_plot(self, plot_name, plot_type, *args, **kwargs):
        self.plots[plot_name] = self.ops[plot_type](plot_name, *args, **kwargs)

    def set_data(self, plot_name, value):
        self.plots[plot_name].set_data(value)

    def update_data(self, plot_name, value):
        self.plots[plot_name].append(value)

    def draw(self):
        for plot_type, plot in self.plots.items():
            plot.draw(self.visdom_logger)
        self.visdom_logger.save([self.env])


class Trainer:
    def __init__(self, epochs,
                 model: torch.nn.Module,
                 loss,
                 val_loss=None,
                 save_fn=None,
                 name='default',
                 img_fn=None,
                 fn_type=None):
        self.epoches = epochs
        self.model = model
        self.loss = loss
        self.val_loss = val_loss or loss
        self.best_metric = None
        self.save_fn = save_fn
        self.name = name
        self.vs = VisdomLogger(name + '_plot')
        self.vs_img = None
        if img_fn:
            self.vs_img = VisdomLogger(name + '_img')
            self.vs_img.add_plot('train_img', fn_type, nrows=4, padding=10)
            self.vs_img.add_plot('val_img', fn_type, nrows=1, padding=10)
        self.img_fn = img_fn

    def _set_data(self, loss_name, stage, value):
        name = f'{loss_name}_{stage}'
        if name not in self.vs.plots:
            self.vs.add_plot(name, 'line')
        self.vs.update_data(name, value)

    def fit(self, train_ds, val_ds, optimizer, scheduler):
        for i in range(self.epoches):
            self.train_epoch(i, train_ds, optimizer)
            self.val_epoch(i, val_ds, scheduler)
            if self.vs_img:
                self.vs.draw()
                self.vs_img.draw()

    def train_epoch(self, epoch, train_ds, optimizer):
        losses = []
        pbar = tqdm.tqdm(train_ds, total=len(train_ds))
        self.model.train(True)
        detailed_info = defaultdict(lambda: [])

        for i, batch in enumerate(pbar):
            batch = {k: v.cuda() if not isinstance(v, list) else v for k, v in batch.items()}
            optimizer.zero_grad()
            pred = self.model(batch)

            loss, details = self.loss(pred, batch)

            loss.backward()
            losses.append(loss)
            for k, v in details.items():
                detailed_info[k].append(v)

            pbar.set_description(f"Train loss:{loss:.7f}")
            pbar.update(1)
            optimizer.step()

            if self.img_fn and i % 150 == 0:
                self.vs_img.set_data('train_img', self.img_fn(batch, pred))
                self.vs_img.draw()

        print(f'Epoch {epoch} Train loss: {torch.tensor(losses).mean():.7f}')
        self._set_data('loss', 'train', torch.tensor(losses).mean())

        for k, v in detailed_info.items():
            ll = torch.tensor(v).mean()
            print(f'Train {k}: {ll:.7f}')
            self._set_data(k, 'train', ll)

    def val_epoch(self, epoch, val_ds, scheduler):
        losses = []
        train_losses = []
        train_detailed_info = defaultdict(lambda: [])
        val_detailed_info = defaultdict(lambda: [])

        pbar = tqdm.tqdm(val_ds, total=len(val_ds))
        self.model.train(False)

        with torch.no_grad():

            for i, batch in enumerate(pbar):
                batch = {k: v.cuda() if not isinstance(v, list) else v for k, v in batch.items()}

                pred = self.model(batch)
                loss, val_details = self.val_loss(pred, batch)
                for k, v in val_details.items():
                    val_detailed_info[k].append(v)

                tloss, tdetails = self.loss(pred, batch)
                for k, v in tdetails.items():
                    train_detailed_info[k].append(v)

                losses.append(loss)
                train_losses.append(tloss)
                pbar.set_description(f"Val loss:{loss:.7f}")
                pbar.update(1)

                if self.img_fn and i % 300 == 0:
                    self.vs_img.set_data('val_img', self.img_fn(batch, pred))
                    self.vs_img.draw()

        metric = torch.tensor(losses).mean()
        loss = torch.tensor(train_losses).mean()

        print(f'Epoch {epoch} Val metrics: {metric:.7f}')
        print(f'Epoch {epoch} Val loss: {loss:.7f}')
        for k, v in val_detailed_info.items():
            ll = torch.tensor(v).mean()
            print(f'Val details {k}: {ll:.7f}')
            self._set_data(k, 'val', ll)

        for k, v in train_detailed_info.items():
            ll = torch.tensor(v).mean()

            print(f'Val/Train details {k}: {ll:.7f}')
            self._set_data(k, 'val', ll)

        scheduler.step(loss)

        if self.best_metric is None or self.best_metric < metric:
            self.save_fn(self.model)
            self.best_metric = metric
            print('Best model updated!')