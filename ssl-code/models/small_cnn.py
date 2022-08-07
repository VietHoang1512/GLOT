from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class Mnist_Duchi(nn.Module):
    # Architecture in Duchi paper
    def __init__(self, num_filters=64, activation=F.relu):
        super(Mnist_Duchi, self).__init__()
        self.l1 = nn.Conv2d(1, num_filters, 8, 2, padding=3)
        self.l2 = nn.Conv2d(num_filters, 2 * num_filters, 6, 2, padding=2)
        self.l3 = nn.Conv2d(2 * num_filters, 2 * num_filters, 5, 1)
        self.l4 = nn.Linear(2 * num_filters * 3 * 3, 10)
        self.act = activation

    def forward(self, x, return_z=False):
        h = self.act(self.l1(x))  # [14,14]
        h = self.act(self.l2(h))  # [7,7]
        h = self.act(self.l3(h))  # [2,2]
        h = torch.flatten(h, 1)
        z = h
        out = self.l4(h)
        if return_z:
            return out, z
        else:
            return out


class Mnist(nn.Module):
    def __init__(self, activation=nn.ReLU(), drop=0.5):
        super(Mnist, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.num_channels, 32, 3)),
                    ("relu1", activation),
                    ("conv2", nn.Conv2d(32, 32, 3)),
                    ("relu2", activation),
                    ("maxpool1", nn.MaxPool2d(2, 2)),  # [14,14]
                    ("conv3", nn.Conv2d(32, 64, 3)),
                    ("relu3", activation),
                    ("conv4", nn.Conv2d(64, 64, 3)),
                    ("relu4", activation),
                    ("maxpool2", nn.MaxPool2d(2, 2)),  # [4, 4]
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(64 * 4 * 4, 200)),
                    ("relu1", activation),
                    ("drop", nn.Dropout(drop)),
                    ("fc2", nn.Linear(200, 200)),
                    ("relu2", activation),
                    ("fc3", nn.Linear(200, self.num_labels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input, return_z=False):
        features = self.feature_extractor(input)
        z = features.view(-1, 64 * 4 * 4)
        logits = self.classifier(z)
        if return_z:
            return logits, z
        else:
            return logits


class Cifar10(nn.Module):
    def __init__(self, activation=nn.ReLU(), drop=0.5):
        super(Cifar10, self).__init__()

        self.num_channels = 3
        self.num_labels = 10

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.num_channels, 64, 3)),
                    ("relu1", activation),
                    ("conv2", nn.Conv2d(64, 64, 3)),
                    ("relu2", activation),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("conv3", nn.Conv2d(64, 128, 3)),
                    ("relu3", activation),
                    ("conv4", nn.Conv2d(128, 128, 3)),
                    ("relu4", activation),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128 * 5 * 5, 256)),
                    ("relu1", activation),
                    ("drop", nn.Dropout(drop)),
                    ("fc2", nn.Linear(256, 256)),
                    ("relu2", activation),
                    ("fc3", nn.Linear(256, self.num_labels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input, return_z=False):
        features = self.feature_extractor(input)
        # print('features_shape:',features.shape)
        z = features.view(-1, 128 * 5 * 5)
        logits = self.classifier(z)
        if return_z:
            return logits, z
        else:
            return logits


class Cifar100(nn.Module):
    def __init__(self, activation=nn.ReLU(), drop=0.5):
        super(Cifar100, self).__init__()

        self.num_channels = 3
        self.num_labels = 100

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(self.num_channels, 64, 3)),
                    ("relu1", activation),
                    ("conv2", nn.Conv2d(64, 64, 3)),
                    ("relu2", activation),
                    ("conv3", nn.Conv2d(64, 64, 3)),
                    ("relu3", activation),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("conv4", nn.Conv2d(64, 128, 3)),
                    ("relu4", activation),
                    ("conv5", nn.Conv2d(128, 128, 3)),
                    ("relu5", activation),
                    ("conv6", nn.Conv2d(128, 128, 3)),
                    ("relu6", activation),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128 * 5 * 5, 256)),
                    ("relu1", activation),
                    ("drop", nn.Dropout(drop)),
                    ("fc2", nn.Linear(256, 256)),
                    ("relu2", activation),
                    ("fc3", nn.Linear(256, 256)),
                    ("relu3", activation),
                    ("fc4", nn.Linear(256, self.num_labels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc4.weight, 0)
        nn.init.constant_(self.classifier.fc4.bias, 0)

    def forward(self, input, return_z=False):
        features = self.feature_extractor(input)
        # print('features_shape:',features.shape)
        z = features.view(-1, 128 * 5 * 5)
        logits = self.classifier(z)
        if return_z:
            return logits, z
        else:
            return logits
