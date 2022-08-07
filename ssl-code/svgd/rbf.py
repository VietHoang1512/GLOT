import numpy as np
import torch
import torch.nn as nn


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

        # if self.sigma is None:
        #   # np_dnorm2 = dnorm2.detach().cpu().numpy()
        #   h = torch.median(dnorm2.detach()) / (2 * torch.log(torch.tensor(X.size(0)) + 1.0))
        #   sigma = torch.sqrt(h)*0.001
        # # else:
        # sigma = self.sigma
        if self.sigma is None:
            sigma = 10 ** int(1 - self.n)
        else:
            sigma = self.sigma
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY
