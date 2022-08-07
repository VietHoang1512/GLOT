#!coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class CNN_block(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, padding, activation):
        super(CNN_block, self).__init__()

        self.act = activation
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size, padding=padding)

        self.bn = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, drop_ratio=0.0):
        super(CNN, self).__init__()

        self.in_plane = 3
        self.out_plane = 128
        self.act = nn.LeakyReLU(0.1)
        self.layer1 = self._make_layer(block, num_blocks[0], 128, 3, padding=1)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(drop_ratio)
        self.layer2 = self._make_layer(block, num_blocks[1], 256, 3, padding=1)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(drop_ratio)
        self.layer3 = self._make_layer(
            block, num_blocks[2], [512, 256, self.out_plane], [3, 1, 1], padding=0
        )
        self.ap3 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.out_plane, self.out_plane // 2)
        self.fc2 = nn.Linear(self.out_plane // 2, num_classes)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.latent_dim = self.out_plane

    def _make_layer(self, block, num_blocks, planes, kernel_size, padding=1):
        if isinstance(planes, int):
            planes = [planes] * num_blocks
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_blocks
        layers = []
        for plane, ks in zip(planes, kernel_size):
            layers.append(block(self.in_plane, plane, ks, padding, self.act))
            self.in_plane = plane
        return nn.Sequential(*layers)

    def forward(self, x, latent=False):
        out = self.layer1(x)
        # print("self.layer1:", out.size())
        out = self.mp1(out)
        # print("self.mp1:", out.size())
        out = self.drop1(out)
        out = self.layer2(out)
        # print("self.layer2:", out.size())
        out = self.mp2(out)
        # print("self.mp2:", out.size())
        out = self.drop1(out)
        out = self.layer3(out)
        # print("self.layer3:", out.size())
        out = self.ap3(out)

        # print("self.ap3:", out.size())
        z = out.view(out.size(0), -1)
        out = F.relu(self.fc1(z))
        out = self.fc2(out)
        if latent:
            return out, z
        return out


def convLarge(num_classes=10, drop_ratio=0.5):
    return CNN(CNN_block, [3, 3, 3], num_classes, drop_ratio)


def test():
    print("--- run conv_large test ---")
    x = torch.randn(2, 3, 32, 32)
    for net in [convLarge(10)]:
        print(net)
        y = net(x)
        print(y.size())


net = convLarge(10)
for name, para in net.named_parameters():
    print(name, para.size())
x = torch.randn(5, 3, 32, 32)
y = net(x)
