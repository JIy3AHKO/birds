from torch import nn
import torch
from torchvision.models import resnet34, resnet18, resnet50


models = {
    'resnet50': (resnet50, 2048),
    'resnet34': (resnet34, 512),
    'resnet18': (resnet18, 512),
}

pools = {
    'max': nn.AdaptiveMaxPool2d,
    'avg': nn.AdaptiveAvgPool2d,
}


class Resnet(nn.Module):

    def __init__(self, num_classes=24, model_type='resnet50', pool='avg'):
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
            nn.Dropout(0.1),
            nn.Linear(self.filters, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, input):
        x = input['x']
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
