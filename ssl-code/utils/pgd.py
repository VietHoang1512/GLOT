from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.models import switch_status


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    n=1,
    step_size=0.003,  # 2/255
    epsilon=0.031,  # 8/255
    perturb_steps=10,
    beta=1.0,
    projecting=True,
    distance="l_inf",
    x_min=0.0,
    x_max=1.0,
    is_runtime=False,
):
    start = datetime.now()
    assert beta == 1.0
    assert distance == "l_inf"
    assert projecting is True
    assert x_max > x_min

    model.eval()
    b, c, w, h = x_natural.shape
    x_adv_particle = torch.zeros_like(x_natural)
    x_adv_particle = x_adv_particle.repeat(n, 1, 1, 1, 1)
    for i in range(n):
        # random initialization
        x_adv = Variable(x_natural.data, requires_grad=True)
        random_noise = (
            torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = nn.CrossEntropyLoss(size_average=False)(
                    model(x_adv), y
                )  # Will not take average over batch

            grad = torch.autograd.grad(loss_ce, [x_adv])[0]  # []
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            if projecting:
                x_adv = torch.min(
                    torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
                )
            x_adv = torch.clamp(x_adv, x_min, x_max)
            # print(loss_ce.item())
        x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
        x_adv_particle[i] = x_adv
    model.train()
    x_adv = x_adv_particle.reshape(-1, c, w, h)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    nat_output = model(x_natural)
    adv_output = model(x_adv)

    loss_natural = F.cross_entropy(nat_output, y, reduction="mean")  # [b,]
    loss_robust = F.cross_entropy(
        adv_output, y.unsqueeze(-1).repeat(n, 1).squeeze(-1), reduction="mean"
    )  # [b, ]

    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    batch_runtime = datetime.now() - start
    if is_runtime:
        loss_natural, loss_robust, loss = 0, 0, 0
        return loss_natural, loss_robust, loss, batch_runtime
    return loss_natural, loss_robust, loss, x_adv


def pgd_attack(model, X, y, device, attack_params, logadv=False, status="train"):
    """
    Reference:
        https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py
        L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
    Args:
        model: pretrained model
        X: input tensor
        y: input target
        attack_params:
            loss_type: 'ce', 'kl' or 'mart'
            epsilon: attack boundary
            step_size: attack step size
            num_steps: number attack step
            order: norm order (norm l2 or linf)
            random_init: random starting point
            x_min, x_max: range of data
    """
    model.eval()

    # assert(attack_params['random_init'] == True)
    # assert(attack_params['projecting'] == True)
    # assert(attack_params['order'] == np.inf)

    X_adv = Variable(X.data, requires_grad=True)

    if attack_params["random_init"]:
        random_noise = (
            torch.FloatTensor(*X_adv.shape)
            .uniform_(-attack_params["epsilon"], attack_params["epsilon"])
            .to(device)
        )
        X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

    X_adves = []
    for _ in range(attack_params["num_steps"]):
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if attack_params["loss_type"] == "ce":
                loss = nn.CrossEntropyLoss()(model(X_adv), y)
            elif attack_params["loss_type"] == "kl":
                loss = nn.KLDivLoss()(
                    F.softmax(model(X_adv), dim=1), F.softmax(model(X), dim=1)
                )

        loss.backward()
        eta = attack_params["step_size"] * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(
            X_adv.data - X.data, -attack_params["epsilon"], attack_params["epsilon"]
        )
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(
            torch.clamp(X_adv, attack_params["x_min"], attack_params["x_max"]),
            requires_grad=True,
        )

        if logadv:
            X_adves.append(X_adv)

    switch_status(model, status)
    X_adv = Variable(X_adv.data, requires_grad=False)
    return X_adv, X_adves


def pgd_attack_Dis(D_model, X, y, device, attack_params, logadv=False, status="train"):
    """
    Reference:
        https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py
        L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
    Args:
        D_model: Discriminator model, output order: (source, classification)
                --> get classification output only
        X: input tensor
        y: input target
        attack_params:
            loss_type: 'ce', 'kl' or 'mart'
            epsilon: attack boundary
            step_size: attack step size
            num_steps: number attack step
            order: norm order (norm l2 or linf)
            random_init: random starting point
            x_min, x_max: range of data
    """
    D_model.eval()

    # assert(attack_params['random_init'] == True)
    # assert(attack_params['projecting'] == True)
    assert attack_params["order"] == np.inf

    X_adv = Variable(X.data, requires_grad=True)

    if attack_params["random_init"]:
        random_noise = (
            torch.FloatTensor(*X_adv.shape)
            .uniform_(-attack_params["epsilon"], attack_params["epsilon"])
            .to(device)
        )
        X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

    X_adves = []
    for _ in range(attack_params["num_steps"]):
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if attack_params["loss_type"] == "ce":
                loss = nn.CrossEntropyLoss()(D_model(X_adv)[1], y)  # CHANGE HERE
            elif attack_params["loss_type"] == "kl":
                loss = nn.KLDivLoss()(
                    F.softmax(D_model(X_adv)[1], dim=1),  # CHANGE HERE
                    F.softmax(D_model(X)[1], dim=1),
                )  # CHANGE HERE

        loss.backward()
        eta = attack_params["step_size"] * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(
            X_adv.data - X.data, -attack_params["epsilon"], attack_params["epsilon"]
        )
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(
            torch.clamp(X_adv, attack_params["x_min"], attack_params["x_max"]),
            requires_grad=True,
        )

        if logadv:
            X_adves.append(X_adv)

    switch_status(D_model, status)
    X_adv = Variable(X_adv.data, requires_grad=False)
    return X_adv, X_adves


def pgd_attack_l2(model, X, y, device, attack_params, logadv=False, status="train"):
    """
    Reference:
        https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py
        L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
    Args:
        model: pretrained model
        X: input tensor
        y: input target
        attack_params:
            loss_type: 'ce', 'kl' or 'mart'
            epsilon: attack boundary
            step_size: attack step size
            num_steps: number attack step
            order: norm order (norm l2 or linf)
            random_init: random starting point
            x_min, x_max: range of data
    """
    model.eval()

    assert attack_params["random_init"] == True
    assert attack_params["projecting"] == True
    assert attack_params["order"] == 2

    delta = torch.zeros_like(X).cuda()

    if attack_params["random_init"]:
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * attack_params["epsilon"]
    delta = clamp(delta, attack_params["x_min"] - X, attack_params["x_max"] - X)

    delta = Variable(delta.data, requires_grad=True)

    X_adves = []
    # Setup optimizers

    for _ in range(attack_params["num_steps"]):
        optimizer_delta = optim.SGD(
            [delta], lr=attack_params["epsilon"] / attack_params["step_size"] * 2
        )

        # optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            loss = (-1) * nn.CrossEntropyLoss()(model(X + delta), y)

        loss.backward()
        # renorming gradient
        grad_norms = delta.grad.view(X.shape[0], -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        optimizer_delta.step()

        # projection
        delta.data.add_(X)
        delta.data.clamp_(0, 1).sub_(X)
        delta.data.renorm_(p=2, dim=0, maxnorm=attack_params["epsilon"])
        X_adv = X + delta
        if logadv:
            X_adves.append(X_adv)

    switch_status(model, status)
    X_adv = Variable(X_adv.data, requires_grad=False)
    return X_adv, X_adves
