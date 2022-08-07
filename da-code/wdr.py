import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class RBF(nn.Module):
    def __init__(self, n, sigma=None):
        super(RBF, self).__init__()

        self.n = n
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
        # if self.sigma is None:
        #     sigma = 10 ** int(1 - self.n)
        # else:
        #     sigma = self.sigma
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def gen_adv(model, x_adv, logits, n, sigma, eps, niter, f):
    # f.write("_" * 50 + "\n")
    kernel = RBF(n, sigma)
    batch_size = x_adv.size(0)
    # calc adversarial perturbation
    x_adv = x_adv.detach()
    for ite in range(niter):
        x_adv.requires_grad_()
        logits_adv = model(x_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits, dim=1),
            reduction="sum",
        ) + F.kl_div(
            F.log_softmax(logits, dim=1),
            F.softmax(logits_adv, dim=1),
            reduction="sum",
        )
        # f.write("iter: {} loss_kl: {}\n".format(ite, loss_kl.item()))
        # f.flush()
        score_func = torch.autograd.grad(loss_kl, [x_adv])[0]

        score_func = score_func.reshape(batch_size, -1)

        K_XX = kernel(x_adv, x_adv.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / batch_size
        # phi = l2_normalize(phi)
        x_adv = (x_adv + eps * phi).detach()
        model.zero_grad()
    return x_adv


if __name__ == "__main__":
    x_adv = torch.randn(40, 32)
