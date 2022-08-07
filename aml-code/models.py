import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from collections import OrderedDict

from small_cnn import * 
from resnet import * 
from wideresnet import * 

# Declare Classifier 
class Toy2D(nn.Module): 
    def __init__(self, nb_classes=3): 
        super(Toy2D, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, nb_classes)
    
    def forward(self, x, return_z=False): 
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        out = self.l4(h)
        if return_z:
            return out, x 
        else:
            return out 

class Toy2D_lamda(nn.Module): 
    def __init__(self): 
        super(Toy2D_lamda, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 1)
    
    def forward(self, x): 
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h 

class GenLamda(nn.Module): 
    def __init__(self, input_dim, num_filters=64, activation=F.relu): 
        super(GenLamda, self).__init__()
        self.l1 = nn.Linear(input_dim, num_filters)
        self.l2 = nn.Linear(num_filters, num_filters)
        self.l3 = nn.Linear(num_filters, 1)
        self.act = activation
    
    def forward(self, x):
        h = self.act(self.l1(x)) # [b,num_filters]
        h = self.act(self.l2(h)) # [b,num_filters]
        out = self.l3(h) # [b,1]
        return out

def activ(key): 
    if key == 'relu': 
        return nn.ReLU() 
    elif key == 'elu': 
        return nn.ELU()

def get_model(ds, model, activation=nn.ReLU()): 
    if ds == 'toy2d': 
        return Toy2D()
    elif ds == 'mnist': 
        # assert (model=='cnn')
        if model == 'cnn':
            return Mnist(activation=activation)
        elif model == 'duchi': 
            return Mnist_Duchi()

    elif ds == 'cifar10': 
        if model == 'cnn': 
            return Cifar10(activation=activation)
        elif model == 'resnet18': 
            return ResNet18()
        elif model == 'wideresnet': 
            return WideResNet()
    elif ds == 'cifar100': 
        if model == 'cnn':
            return Cifar100(activation=activation)
        elif model == 'resnet18': 
            return ResNet18(num_classes=100)

def adjust_learning_rate_mnist(optimizer, epoch, lr):
    """decrease the learning rate"""
    _lr = lr 
    if epoch >= 55:
        _lr = lr * 0.1
    if epoch >= 75:
        _lr = lr * 0.01
    if epoch >= 90:
        _lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return optimizer

def adjust_learning_rate_cifar10(optimizer, epoch, lr):
    """decrease the learning rate"""
    _lr = lr
    if epoch >= 100:
        _lr = lr * 0.1
    if epoch >= 105:
        _lr = lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return optimizer

def adjust_learning_rate(optimizer, epoch, lr, ds): 
    if ds == 'mnist': 
        return adjust_learning_rate_mnist(optimizer, epoch, lr)
    elif ds in ['cifar10', 'cifar100']: 
        return adjust_learning_rate_cifar10(optimizer, epoch, lr)


def get_optimizer(ds, model, architecture):
    if ds == 'mnist': 
        if architecture == 'duchi': 
            lr = 0.001
            opt = optim.Adam(model.parameters(), lr=lr)
        elif architecture == 'cnn': 
            lr = 0.01
            momentum = 0.9
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)             

    elif ds == 'cifar10': 
        if architecture == 'cnn':
            lr = 0.001
            opt = optim.Adam(model.parameters(), lr=lr)
        
        else:
            lr = 0.1
            momentum = 0.9
            weight_decay = 5e-4
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)            
    
    elif ds == 'cifar100': 
        if architecture == 'cnn':
            lr = 0.001
            opt = optim.Adam(model.parameters(), lr=lr)
        
        elif architecture == 'resnet18': 
            lr = 0.1
            momentum = 0.9
            weight_decay = 5e-4
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)            

    else: 
        raise ValueError

    return opt, lr

def switch_status(model, status): 
    if status == 'train': 
        model.train()
    elif status == 'eval': 
        model.eval()
    else: 
        raise ValueError
#------------------------------------------------------
class DataWithIndex(Dataset):
    def __init__(self, train_data):
        self.data = train_data
        
    def __getitem__(self, index):
        data, target = self.data[index]
        
        return data, target, index

    def __len__(self):
        return len(self.data)