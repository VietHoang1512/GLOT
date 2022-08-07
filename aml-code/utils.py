import argparse
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

# Create heatmap data 
class Grid(Dataset):
    def __init__(self, train_x, scale=1.0, n_test=100):
        delta = np.max(train_x[:,0]) - np.min(train_x[:,0])
        low = np.min(train_x[:,0]) - delta*scale
        high = np.max(train_x[:,0]) + delta*scale

        a = np.arange(low,high,(high-low)/n_test)
        x_1,x_2 = np.meshgrid(a,a)
        t_1 = np.reshape(x_1, [-1,])
        t_2 = np.reshape(x_2, [-1,])
        x_data = np.zeros(shape=[n_test**2,2])
        x_data[:,0] = t_1[:n_test**2]
        x_data[:,1] = t_2[:n_test**2]
        self.data = torch.tensor(x_data, dtype=torch.float32) 
        self.x_1 = x_1 
        self.x_2 = x_2 
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

# Get heatmap 
def onehot2score(onehot): 
    n,d = np.shape(onehot)
    x = np.zeros(shape=[n,d])
    for i in range(d): 
        x[:,i] = onehot[:,i]*(i+1)
    s = np.sum(x, axis=1)
    return s 


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                        num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def label_smoothing(y_batch_tensor, num_classes, delta):
    '''
        y_smth = 
        delta = 0 --> y_smth = y 
        delta = 1 --> y_smth = 1/(N-1) * [1,1,1,0,1] where 0 is true class
    '''
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * y_batch_tensor + delta / (num_classes - 1)
    return y_batch_smooth


class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss

def get_pred(model, data_loader, device): 
    model.eval()
    result = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            output = torch.nn.Softmax()(output)
            result.append(output.cpu().numpy())

    result = np.concatenate(result, axis=0)
    return result

def get_acc(output, target): 
    pred = output.argmax(dim=1, keepdim=True)
    acc = torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor))
    return acc 

class CrossEntropyWithLabelSmoothing(nn.Module): 
    def __init__(self, reduce=True, num_classes=10): 
        super(CrossEntropyWithLabelSmoothing, self).__init__()
        self.reduce = reduce 
        self.num_classes = num_classes
    
    def one_hot_tensor(self, y_batch_tensor, num_classes):
        y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0)
        y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
        return y_tensor

    def label_smoothing(self, y_batch_tensor, num_classes, delta):
        '''
            y_smth = 
            delta = 0 --> y_smth = y 
            delta = 1 --> y_smth = 1/(N-1) * [1,1,1,0,1] where 0 is true class
        '''
        y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * y_batch_tensor + delta / (num_classes - 1)
        return y_batch_smooth

    def forward(self, inputs, targets, delta=0.0): 
        y_gt = self.one_hot_tensor(targets, self.num_classes)
        y_sm = self.label_smoothing(y_gt, self.num_classes, delta=delta)
        y_sm = y_sm.detach() # IMPORTANT 

        log_likelihood = -F.log_softmax(inputs, dim=1)
        temp = torch.mul(log_likelihood, y_sm) # shape [b, num_classes]

        if self.reduce in [True, 'mean']:
            "Reduce='mean', average the loss over batch"
            loss = torch.mean(torch.sum(temp, dim=1), dim=0)
        elif self.reduce in [False, 'sum']: 
            loss = torch.sum(torch.sum(temp, dim=1), dim=0)
        else: 
            raise ValueError
        return loss 