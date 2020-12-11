from functools import lru_cache

from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18, resnet50, densenet121
from efficientnet_pytorch import EfficientNet

from misc import spec_augment
from pann import Cnn14_DecisionLevelAtt
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


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn_att = nn.BatchNorm1d(n_out)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.cla(x)
        x = self.bn_att(norm_att * cla)

        return x


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

    def __init__(self,
                 num_classes=24,
                 model_type='resnet50',
                 pool='avg',
                 dropout=0.4,
                 freq_drop=0.15,
                 time_drop=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.adapt_pool = pools[pool]
        d, self.filters = models[model_type]
        d = d(pretrained=True)
        self.conv1 = d.conv1
        self.bn1 = d.bn1
        self.relu = d.relu
        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=1)
        self.maxpool = d.maxpool

        self.dropout = dropout

        self.interpolate_ratio = 32
        self.layer1 = d.layer1
        self.layer2 = d.layer2
        self.layer3 = d.layer3
        self.layer4 = d.layer4

        self.att = AttBlock(self.filters, self.filters)

        self.regressor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.filters, num_classes),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(self.filters * 4, self.filters // 2, 1),
            nn.BatchNorm1d(self.filters // 2),
            nn.LeakyReLU()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(self.filters, self.filters // 2, 3, padding=1),
            nn.BatchNorm1d(self.filters // 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.filters // 2, num_classes, 1, padding=1),

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

        self.freq_drop = freq_drop
        self.time_drop = time_drop

    def forward(self, input):
        waveform = input['x']

        spec = self.transforms(waveform) / 30
        #
        if self.training:
            spec = spec_augment(spec,
                                freq_masking_max_percentage=self.freq_drop,
                                time_masking_max_percentage=self.time_drop)

        frames_num = spec.shape[3]
        grid = get_coordinate_grid(spec.shape, spec.device)
        x = torch.cat([spec, spec, grid], dim=1)
        bs = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = x.view(bs, self.filters * 4, -1)

        x = torch.mean(x, dim=2)
        x = F.dropout(x, p=self.dropout, training=self.training)
      #  x = self.att(x)
       #  pooled_x = nn.AdaptiveMaxPool1d(1)(x).view(-1, self.filters)
        # clipwise_output = self.regressor(pooled_x)
        segmentwise_output = self.proj(x)
        segmentwise_output = segmentwise_output # .transpose(1, 2)

        # # Get framewise output
        # framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # print(framewise_output.shape)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)
        clipwise_output = nn.AdaptiveMaxPool1d(1)(segmentwise_output).view(-1, self.num_classes)

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


def get_model(name, **kwargs):
    if 'resnet' in name:
        return Resnet(model_type=name,  **kwargs)
    elif 'densenet' in name:
        return Densenet( **kwargs)
    elif 'efficientnet' in name:
        return Effnet(model_type=name, **kwargs)
    elif 'cnn14' in name:
        return Cnn14()
    else:
        raise ValueError(f"not supported model {name}")
