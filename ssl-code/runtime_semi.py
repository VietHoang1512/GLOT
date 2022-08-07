import argparse
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from models.models import get_model, get_optimizer
from svgd.semi import trades_svgd_supsem_loss
from utils.datasets import get_data_loader, get_supsem_settings
from utils.vat_new import vat_loss

torch.manual_seed(2021)
np.random.rand(2021)

# 1. all particles 2. sequence 3. noises

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--model", type=str, default="cnn13", help="cnn|resnet")

parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--num_label", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--trial", type=int, default=2, help="4000")
parser.add_argument("--gpu_id", type=str, default="1")

parser.add_argument("--method", type=str, default="trades_svgd_mixup")
parser.add_argument("--n", type=int, default=4, help="num particles")


opt = parser.parse_args()


dataset = "cifar10"
attacks = ["vat", "trades_svgd"]


opt.dataset = dataset

particles = [1, 2, 4]
model = "cnn13"

opt.model = model
for attack in attacks:
    opt.method = attack
    for n in particles:

        opt.n = n

        params = get_supsem_settings(opt.dataset)
        params["num_label"] = opt.num_label
        label_loader, unlab_loader, test_loader = get_data_loader(
            ds=opt.dataset, batch_size=opt.batch_size, num_label=opt.num_label
        )

        def train(model, label_x, label_y, unlab_x, optimizer, epoch, is_runtime=True):

            if opt.method == "trades_svgd":
                loss_natural, loss_robust, loss, x_adv = trades_svgd_supsem_loss(
                    model,
                    label_x,
                    label_y,
                    unlab_x,
                    optimizer,
                    n=opt.n,
                    sigma=None,
                    xi=params["xi"],
                    eps=params["epsilon"],
                    niter=params["perturb_steps"],
                    beta=params["beta"],
                    epoch=epoch,
                    is_runtime=is_runtime,
                )
            elif opt.method == "vat":
                # vat_loss = VATLoss(xi=10.0, eps=params["epsilon"], ip=1, beta=params["beta"])
                loss_natural, loss_robust, loss, x_adv = vat_loss(
                    model,
                    label_x,
                    label_y,
                    unlab_x,
                    optimizer,
                    n=opt.n,
                    xi=params["xi"],
                    eps=params["epsilon"],
                    niter=params["perturb_steps"],
                    beta=params["beta"],
                    epoch=epoch,
                    is_runtime=is_runtime,
                )
            else:
                loss_robust = torch.tensor(0.0).detach()
                ce = nn.CrossEntropyLoss()
                y_pred = model(label_x)
                loss_natural = ce(y_pred, label_y)
                loss = loss_natural

            return loss_natural, loss_robust, x_adv

        model = get_model(ds=opt.dataset, model=opt.model, activation=nn.ReLU())
        model.cuda()

        optimizer, lr = get_optimizer(
            ds=opt.dataset, model=model, architecture=opt.model
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.num_epochs, eta_min=0.0001
        )
        print("#" * 10)
        start_time = datetime.now()
        scheduler.step()

        for i, ((label_x, label_y), (unlab_x, unlab_y)) in enumerate(
            zip(cycle(label_loader), unlab_loader)
        ):
            label_x, label_y = label_x.cuda(), label_y.cuda()
            unlab_x, unlab_y = unlab_x.cuda(), unlab_y.cuda()

            train(model, label_x, label_y, unlab_x, optimizer, 1, is_runtime=False)
        batch_runtime = datetime.now() - start_time
        total_time = batch_runtime.total_seconds()
        print(
            opt.dataset,
            opt.method,
            opt.model,
            opt.n,
            total_time,
            total_time / ((i + 1) * opt.batch_size),
        )
