from torch import nn
import torch
from torchvision.models import resnet34, resnet18, resnet50, densenet121
from efficientnet_pytorch import EfficientNet
from panns_inference.models import Cnn14
import torchaudio as ta

models = {
    'resnet50': (resnet50, 2048),
    'resnet34': (resnet34, 512),
    'resnet18': (resnet18, 512),
}

pools = {
    'max': nn.AdaptiveMaxPool2d,
    'avg': nn.AdaptiveAvgPool2d,
}


class Densenet(nn.Module):
    def __init__(self, num_classes=24, pool='avg', dropout=0.4):
        super().__init__()

        self.d = densenet121(pretrained=True)
        self.filters = 1024
        self.adapt_pool = pools[pool]

        self.regressor = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.filters, num_classes),
        )

    def forward(self, x):
        x = x['x']
        x = torch.cat([x, x, x], dim=1)
        x = self.d.features(x)
        x = torch.nn.functional.relu(x, inplace=True)

        features = self.adapt_pool(1)(x).view(-1, self.filters)
        features = features.view(x.size(0), -1)

        res = self.regressor(features)

        return {'y': res}


class Resnet(nn.Module):

    def __init__(self, num_classes=24, model_type='resnet50', pool='avg', dropout=0.4):
        super().__init__()
        self.adapt_pool = pools[pool]
        d, self.filters = models[model_type]
        d = d(pretrained=True)
        self.conv1 = d.conv1
        self.bn1 = d.bn1
        self.relu = d.relu
        self.maxpool = d.maxpool

        self.layer1 = d.layer1
        self.layer2 = d.layer2
        self.layer3 = d.layer3
        self.layer4 = d.layer4

        self.regressor = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.filters, num_classes),
        )

        self.transforms = nn.Sequential(
            ta.transforms.MelSpectrogram(
                sample_rate=48000,
                n_fft=1024 * 2,
                win_length=1024 * 2,
                hop_length=1024 * 2,
                f_min=50,
                f_max=14000,
                n_mels=64
            ),
            ta.transforms.AmplitudeToDB(top_db=80),
        )

    def forward(self, input):
        x = input['x']
        x = self.transforms(x)
        x = torch.cat([x, x, x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.adapt_pool(1)(x).view(-1, self.filters)
        features = features.view(x.size(0), -1)

        res = self.regressor(features)

        return {'y': res}


class Effnet(nn.Module):
    def __init__(self, num_classes=24, model_type="efficientnet-b0", pool='avg', dropout=0.4):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(model_type, advprop=True)
        self.adapt_pool = pools[pool]
        self.filters = 1280
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.filters, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x['x']
        x = torch.cat([x, x, x], dim=1)
        x = self.effnet.extract_features(x)

        features = self.adapt_pool(1)(x).view(-1, self.filters)
        features = features.view(x.size(0), -1)

        res = self.regressor(features)

        return {'y': res}


def get_model(name, dropout):
    if 'resnet' in name:
        return Resnet(model_type=name, dropout=dropout)
    elif 'densenet' in name:
        return Densenet(dropout=dropout)
    elif 'efficientnet' in name:
        return Effnet(model_type=name, dropout=dropout)
    else:
        raise ValueError(f"not supported model {name}")
