from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    n=1,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    projecting=True,
    distance="l_inf",
    x_min=0.0,
    x_max=1.0,
    is_runtime=False,
):
    # define KL-loss
    start = datetime.now()
    assert projecting is True
    assert distance == "l_inf"
    assert x_max > x_min

    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size, c, w, h = x_natural.shape
    x_adv_particle = torch.zeros_like(x_natural)
    x_adv_particle = x_adv_particle.repeat(n, 1, 1, 1, 1)
    for i in range(n):
        # generate adversarial example
        x_adv = (
            x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        )
        if distance == "l_inf":
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(
                        F.log_softmax(model(x_adv), dim=1),
                        F.softmax(model(x_natural), dim=1),
                    )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                if projecting:
                    x_adv = torch.min(
                        torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
                    )
                x_adv = torch.clamp(x_adv, x_min, x_max)

        else:
            x_adv = torch.clamp(x_adv, x_min, x_max)
        x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
        x_adv_particle[i] = x_adv

    model.train()

    x_adv = x_adv_particle.reshape(-1, c, w, h)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(x_natural.repeat(n, 1, 1, 1)), dim=1),
    )
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    batch_runtime = datetime.now() - start
    if is_runtime:
        loss_natural, loss_robust, loss = 0, 0, 0
        return loss_natural, loss_robust, loss, batch_runtime
    return loss_natural, loss_robust, loss, x_adv
