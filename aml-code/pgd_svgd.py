import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import switch_status
from utils import CrossEntropyWithLabelSmoothing 

class RBF(nn.Module):
    def __init__(self, num_particles, sigma=None):
        super(RBF, self).__init__()

        self.num_particles = num_particles
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        if self.sigma is None:
            # np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = torch.median(dnorm2.detach()) / (
                2 * torch.log(torch.tensor(X.size(0)) + 1.0)
            )
            sigma = torch.sqrt(h) * 0.001
        else:
            sigma = self.sigma
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd_svgd_symkl(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf',
                keep_prob=0.5, 
                x_max=1.0, 
                x_min=0.0, 
                num_particles=2, 
                sigma=None, 
                ):
    # assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    del keep_prob
    # model.eval()
    assert(model.training is True)

    kernel = RBF(num_particles, sigma)
    batch_size, c, w, h = x_natural.shape # batch_size before repeat

    if num_particles > 1: 
        x_natural = x_natural.repeat(num_particles, 1, 1, 1)
        y = y.unsqueeze(-1).repeat(num_particles, 1).squeeze(-1)

    x_natural = torch.reshape(x_natural, [num_particles * batch_size, -1])

    def _reshape(X_flat): 
        return torch.reshape(X_flat, [num_particles*batch_size, c, w, h])

    # random initialization 
    x_adv = Variable(x_natural.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    with torch.no_grad():
        y_pred = model(_reshape(x_natural))

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            output = model(_reshape(x_adv))
            loss = 0.5 * nn.KLDivLoss(size_average=False)(
                        F.log_softmax(output, dim=1),
                        F.softmax(y_pred, dim=1),
                    ) + 0.5 * nn.KLDivLoss(size_average=False)(
                        F.log_softmax(y_pred, dim=1),
                        F.softmax(output, dim=1),
                    )

        score_func = torch.autograd.grad(loss, [x_adv])[0]
        K_XX = kernel(x_adv, x_adv.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]
        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size*num_particles)
        x_adv = x_adv.detach() + step_size * torch.sign(phi.detach())

        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)

    model.train()

    x_adv = _reshape(x_adv)
    x_natural = _reshape(x_natural)

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    return x_adv, x_natural

def pgd_svgd_ce(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf',
                keep_prob=0.5, 
                x_max=1.0, 
                x_min=0.0, 
                num_particles=2, 
                sigma=None, 
                ):
    # assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    del keep_prob
    # model.eval()
    assert(model.training is True)

    kernel = RBF(num_particles, sigma)
    batch_size, c, w, h = x_natural.shape # batch_size before repeat

    if num_particles > 1: 
        x_natural = x_natural.repeat(num_particles, 1, 1, 1)
        y = y.unsqueeze(-1).repeat(num_particles, 1).squeeze(-1)

    x_natural = torch.reshape(x_natural, [num_particles * batch_size, -1])

    def _reshape(X_flat): 
        return torch.reshape(X_flat, [num_particles*batch_size, c, w, h])

    # random initialization 
    x_adv = Variable(x_natural.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    with torch.no_grad():
        y_pred = model(_reshape(x_natural))

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            output = model(_reshape(x_adv))
            loss = nn.CrossEntropyLoss(size_average=False)(output, y)

        score_func = torch.autograd.grad(loss, [x_adv])[0]
        K_XX = kernel(x_adv, x_adv.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]
        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size*num_particles)
        x_adv = x_adv.detach() + step_size * torch.sign(phi.detach())

        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)

    model.train()

    x_adv = _reshape(x_adv)
    x_natural = _reshape(x_natural)

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    return x_adv, x_natural

def pgd_svgd_kl(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                alpha=1.0,
                beta=1.0,
                projecting=True,
                distance='l_inf',
                keep_prob=0.5, 
                x_max=1.0, 
                x_min=0.0, 
                num_particles=2, 
                sigma=None, 
                ):
    # assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    del keep_prob
    # model.eval()
    assert(model.training is True)

    kernel = RBF(num_particles, sigma)
    batch_size, c, w, h = x_natural.shape # batch_size before repeat

    if num_particles > 1: 
        x_natural = x_natural.repeat(num_particles, 1, 1, 1)
        y = y.unsqueeze(-1).repeat(num_particles, 1).squeeze(-1)

    x_natural = torch.reshape(x_natural, [num_particles * batch_size, -1])

    def _reshape(X_flat): 
        return torch.reshape(X_flat, [num_particles*batch_size, c, w, h])

    # random initialization 
    x_adv = Variable(x_natural.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    with torch.no_grad():
        y_pred = model(_reshape(x_natural))

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            output = model(_reshape(x_adv))
            loss = nn.KLDivLoss(size_average=False)(
                        F.log_softmax(output, dim=1),
                        F.softmax(y_pred, dim=1),
                    )

        score_func = torch.autograd.grad(loss, [x_adv])[0]
        K_XX = kernel(x_adv, x_adv.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]
        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size*num_particles)
        x_adv = x_adv.detach() + step_size * torch.sign(phi.detach())

        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)

    model.train()

    x_adv = _reshape(x_adv)
    x_natural = _reshape(x_natural)

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    return x_adv, x_natural