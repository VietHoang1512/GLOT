import contextlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from potential_net import EntropicWassersteinLoss, pairwise_forward_kl
from svgd.rbf import RBF


@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def exp_rampup(rampup_length=30):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper


def l2_normalize(d):
    d_abs_max = torch.max(torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
        d.size(0), 1, 1, 1
    )
    d /= 1e-12 + d_abs_max
    d /= torch.sqrt(
        1e-6
        + torch.sum(torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True)
    )
    return d


# def mse_with_softmax(logit1, logit2):
#     assert logit1.size() == logit2.size()
#     return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))


def gen_r_vadv(model, x, vlogits, n, sigma, xi, eps, niter):
    kernel = RBF(n, sigma)
    batch_size, c, w, h = x.size()

    # perpare random unit tensor
    d = torch.rand(x.shape).to(x.device)
    d = l2_normalize(d)
    # calc adversarial perturbation
    for _ in range(niter):
        d = d.reshape(batch_size, c, w, h)
        d.requires_grad_()
        with torch.enable_grad():
            rlogits = model(x + xi * d)
        # adv_dist = mse_with_softmax(rlogits, vlogits)
        adv_dist = nn.KLDivLoss()(
            F.log_softmax(rlogits, dim=1),
            F.softmax(vlogits, dim=1),
        ) + nn.KLDivLoss()(
            F.log_softmax(vlogits, dim=1),
            F.softmax(rlogits, dim=1),
        )
        score_func = torch.autograd.grad(adv_dist.sum(), [d])[0]

        d = d.reshape(batch_size, -1)
        score_func = score_func.reshape(batch_size, -1)

        K_XX = kernel(d, d.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), d)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / (batch_size)
        phi = phi.reshape(batch_size, c, w, h)
        phi = l2_normalize(phi)
    return eps * phi.detach()


def trades_svgd(
    model,
    label_x,
    label_y,
    unlab_x,
    optimizer,
    n,
    sigma,
    xi,
    eps,
    niter,
    beta,
    epoch,
    is_runtime=False,
):
    model.train(True)
    ce = nn.CrossEntropyLoss()
    with torch.enable_grad():
        start = datetime.now()
        ##=== forward ===
        outputs = model(label_x)
        loss_natural = ce(outputs, label_y)
        ##=== Semi-supervised Training ===
        ## local distributional smoothness (LDS)
        unlab_x = unlab_x.repeat(n, 1, 1, 1)

        unlab_outputs = model(unlab_x)
        with torch.no_grad():
            vlogits = unlab_outputs.clone().detach()

        with disable_tracking_bn_stats(model):
            r_vadv = gen_r_vadv(model, unlab_x, vlogits, n, sigma, xi, eps, niter)
            x_adv = unlab_x + r_vadv
            rlogits = model(x_adv)
            loss_robust = nn.KLDivLoss()(
                F.log_softmax(rlogits, dim=1),
                F.softmax(vlogits, dim=1),
            ) + nn.KLDivLoss()(
                F.log_softmax(vlogits, dim=1),
                F.softmax(rlogits, dim=1),
            )
            loss_robust *= exp_rampup()(epoch) * beta

            loss = loss_natural + loss_robust
        ## backwark
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_runtime = datetime.now() - start
        if is_runtime:
            loss_natural, loss_robust, loss = 0, 0, 0
            return loss_natural, loss_robust, loss, batch_runtime
    return loss_natural, loss_robust, loss, x_adv


def trades_svgd_ot(
    model,
    potential,
    label_x,
    label_y,
    unlab_x,
    optimizer,
    potential_optimizer,
    n,
    sigma,
    xi,
    eps,
    niter,
    beta,
    gamma,
    epoch,
):
    model.train(True)
    ce = nn.CrossEntropyLoss()
    with torch.enable_grad():
        lbs, ubs = label_x.size(0), unlab_x.size(0)

        ##=== forward ===
        outputs, z = model(label_x, latent=True)
        loss_natural = ce(outputs, label_y)

        unlab_x = unlab_x.repeat(n, 1, 1, 1)

        unlab_outputs, unlab_z = model(unlab_x, latent=True)
        with torch.no_grad():
            vlogits = unlab_outputs.clone().detach()

        C = torch.cdist(z, unlab_z[:ubs], p=2) + 0.5 * pairwise_forward_kl(
            outputs, unlab_outputs[:ubs]
        )
        loss_ws = EntropicWassersteinLoss(z, C, potential, 0.1)

        loss_ws *= exp_rampup()(epoch)
        with disable_tracking_bn_stats(model):
            r_vadv = gen_r_vadv(model, unlab_x, vlogits, n, sigma, xi, eps, niter)
            x_adv = unlab_x + r_vadv
            rlogits = model(x_adv)
            loss_robust = nn.KLDivLoss()(
                F.log_softmax(rlogits, dim=1),
                F.softmax(vlogits, dim=1),
            ) + nn.KLDivLoss()(
                F.log_softmax(vlogits, dim=1),
                F.softmax(rlogits, dim=1),
            )
            loss_robust *= exp_rampup()(epoch) * beta

            loss = loss_natural + loss_robust + gamma * loss_ws
        ## backwark

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        potential.zero_grad()

        z = z.detach().clone()
        unlab_z = unlab_z[:ubs].detach().clone()
        outputs = outputs.detach().clone()
        unlab_outputs = unlab_outputs[:ubs].detach().clone()

        C = torch.cdist(z, unlab_z, p=2) + 0.5 * pairwise_forward_kl(
            outputs, unlab_outputs
        )
        neg_loss_ws = -EntropicWassersteinLoss(z, C, potential, 0.1)

        neg_loss_ws.backward()
        potential_optimizer.step()
        potential.zero_grad()

    return loss_natural, loss_robust, loss_ws, loss
