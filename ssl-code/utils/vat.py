import contextlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def mse_with_softmax(logit1, logit2):
    assert logit1.size() == logit2.size()
    return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))


def gen_r_vadv(model, x, vlogits, xi, eps, niter):

    # perpare random unit tensor
    d = torch.rand(x.shape).to(x.device)
    d = l2_normalize(d)
    # calc adversarial perturbation
    for _ in range(niter):
        d.requires_grad_()
        rlogits = model(x + xi * d)
        adv_dist = mse_with_softmax(rlogits, vlogits)
        adv_dist.backward()
        d = l2_normalize(d.grad)
        model.zero_grad()
    return eps * d.detach()


def vat_loss(
    model,
    label_x,
    label_y,
    unlab_x,
    optimizer,
    n,
    xi,
    eps,
    niter,
    beta,
    epoch,
    is_runtime=False,
):
    start = datetime.now()
    model.train(True)
    ce = nn.CrossEntropyLoss()
    with torch.enable_grad():
        lbs, ubs = label_x.size(0), unlab_x.size(0)

        ##=== forward ===
        outputs = model(label_x)
        loss_natural = ce(outputs, label_y)

        ##=== Semi-supervised Training ===
        ## local distributional smoothness (LDS)
        unlab_outputs = model(unlab_x)
        with torch.no_grad():
            vlogits = unlab_outputs.clone().detach()

        loss_robust = 0.0
        with disable_tracking_bn_stats(model):
            for _ in range(n):
                r_vadv = gen_r_vadv(model, unlab_x, vlogits, xi, eps, niter)
                x_adv = unlab_x + r_vadv
                rlogits = model(x_adv)
                loss_robust += mse_with_softmax(rlogits, vlogits)
            loss_robust *= exp_rampup()(epoch) * beta / n

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
