from functools import lru_cache

from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18, resnet50, densenet121
from efficientnet_pytorch import EfficientNet
from pann import Cnn14_DecisionLevelAtt, AttBlock
import torchaudio as ta

from pytorch_utils import interpolate, pad_framewise_output

models = {
    'resnet50': (resnet50, 2048),
    'resnet34': (resnet34, 512),
    'resnet18': (resnet18, 512),
}

pools = {
    'max': nn.AdaptiveMaxPool2d,
    'avg': nn.AdaptiveAvgPool2d,
}


@lru_cache(maxsize=8)
def get_coordinate_grid(shape, device):
    g = torch.arange(shape[2], requires_grad=False).float()
    g = g.repeat(shape[3]).view(1, 1, shape[3], shape[2]).transpose(3, 2)
    g /= shape[2]
    g = g.repeat(shape[0], 1, 1, 1)
    return g.to(device)


class Cnn14(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Cnn14_DecisionLevelAtt(48000, 1024 * 2, 1024 * 2, 64, 50, 14000, 24)

    def forward(self, x):
        x = self.m(x['x'])
        x['y'] = x['clipwise_output']

        return x

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
        grid = get_coordinate_grid(x.shape, x.device)
        x = torch.cat([x, x, grid], dim=1)
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

        self.interpolate_ratio = 32
        self.layer1 = d.layer1
        self.layer2 = d.layer2
        self.layer3 = d.layer3
        self.layer4 = d.layer4

        self.regressor = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.filters, num_classes),
        )

        self.fc1 = nn.Linear(self.filters, self.filters)
        self.att_block = AttBlock(self.filters, 24, activation='linear')

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
        waveform = input['x']

        spec = self.transforms(waveform)
        frames_num = spec.shape[3]
        grid = get_coordinate_grid(spec.shape, spec.device)

        x = torch.cat([spec, spec, grid], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, _, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output # .transpose(1, 2)

        # # Get framewise output
        # framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # print(framewise_output.shape)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_output = F.interpolate(segmentwise_output, frames_num, mode='linear', align_corners=True)

        output_dict = {'framewise_output': framewise_output,
                       'clipwise_output': clipwise_output,
                       'spec': spec}

        return output_dict


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
    elif 'cnn14' in name:
        return Cnn14()
    else:
        raise ValueError(f"not supported model {name}")
