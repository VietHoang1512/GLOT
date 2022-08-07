import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from distance import distance

def get_positive_mask(labels): 
    # ATTENTION HERE, positive mask will ignore the diagonal
    # requires one_hot labels 
    # l = torch.matmul(labels, labels.T) # [2b,2b]
    # m = torch.ones_like(l).fill_diagonal_(0) # [2b,2b]
    # pos_mask = torch.multiply(l, m) # [2b,2b]
    labels = labels.type(torch.cuda.FloatTensor)
    l = torch.matmul(labels, labels.T).fill_diagonal_(0) # [2b,2b]
    return l

def compact_loss(latents, labels, num_classes, dist='cosine'): 
    if len(labels.shape) == 1 or labels.shape[-1] == 1: 
        labels = F.one_hot(labels, num_classes)
    
    pos_mask = get_positive_mask(labels)
    loss = distance(latents, latents, dist=dist, pairwise=True)
    assert(loss.shape == pos_mask.shape)
    loss = torch.mean(pos_mask * loss) 

    return loss 