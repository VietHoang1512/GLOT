import argparse
import os
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models.models import get_model, get_optimizer
from potential_net import KanNet
from svgd.semi import trades_svgd, trades_svgd_ot
from utils.accuracy import accuracy
from utils.datasets import get_data_loader
from utils.vat import vat_loss


def write_log(log, log_path):
    f = open(log_path, mode="a")
    f.write(str(log))
    f.write("\n")
    f.close()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_supsem_settings(ds):

    if ds == "mnist":
        xi = 10.0
        epsilon = 0.01
        step_size = 0.01
        perturb_steps = 1
        beta = 30.0
        num_label = 500
        params = {
            "epsilon": epsilon,
            "step_size": step_size,
            "perturb_steps": perturb_steps,
            "beta": beta,
            "num_label": num_label,
            "xi": xi,
        }
        return params

    elif ds == "cifar10":
        xi = 10.0
        epsilon = 0.0005
        step_size = 0.007
        perturb_steps = 1
        beta = 30.0
        num_label = 4000
        params = {
            "epsilon": epsilon,
            "step_size": step_size,
            "perturb_steps": perturb_steps,
            "beta": beta,
            "num_label": num_label,
            "xi": xi,
        }
        return params


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--model", type=str, default="cnn13", help="cnn|resnet")

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_label", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=100)

parser.add_argument("--method", type=str, default="trades_svgd")
parser.add_argument("--beta", type=float, default=30.0, help="wdr weight")
parser.add_argument("--gamma", type=float, default=0.0, help="ws weight")
parser.add_argument("--n", type=int, default=4, help="num particles")
parser.add_argument("--seed", type=int, default=0, help="seed")

opt = parser.parse_args()

seed_everything(opt.seed)
print(opt)
paras = (
    str(opt)
    .replace(", ", "")
    .replace("'", "")
    .replace("(", "")
    .replace(")", "")
    .replace("Namespace", "")
    .replace(f"method={opt.method}", "")
    .replace(f"model={opt.model}", "")
    .replace(f"dataset={opt.dataset}", "")
    .replace(f"num_label={opt.num_label}", "")
)

print(paras)
prefix = [opt.dataset, opt.method, opt.model, str(opt.num_label), paras]
log = os.path.join("log_semi", *prefix)

# os.system("rm -Rf {}".format(log))
writer = SummaryWriter(log)
log_file = os.path.join(log, "log.txt")
with open(os.path.join(log, "config.yaml"), "w") as outfile:
    yaml.dump(vars(opt), outfile, default_flow_style=False)
params = get_supsem_settings(opt.dataset)
params["num_label"] = opt.num_label
params["beta"] = opt.beta
params["gamma"] = opt.gamma

print(params)
label_loader, unlab_loader, test_loader = get_data_loader(
    ds=opt.dataset, batch_size=opt.batch_size, num_label=opt.num_label
)


def train(model, kannet, label_x, label_y, unlab_x, optimizer, kannet_optimizer, epoch):
    loss_ws = torch.zeros(1)
    if opt.method == "trades_svgd":
        loss_natural, loss_robust, loss, _ = trades_svgd(
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
        )
    if opt.method == "trades_svgd_ot":
        loss_natural, loss_robust, loss_ws, loss = trades_svgd_ot(
            model,
            kannet,
            label_x,
            label_y,
            unlab_x,
            optimizer,
            kannet_optimizer,
            n=opt.n,
            sigma=None,
            xi=params["xi"],
            eps=params["epsilon"],
            niter=params["perturb_steps"],
            beta=params["beta"],
            gamma=params["gamma"],
            epoch=epoch,
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
        )
    else:
        loss_robust = torch.tensor(0.0).detach()
        ce = nn.CrossEntropyLoss()
        y_pred = model(label_x)
        loss_natural = ce(y_pred, label_y)
        loss = loss_natural
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_natural, loss_robust, loss_ws


model = get_model(ds=opt.dataset, model=opt.model, activation=nn.ReLU())
model.cuda()
kannet = KanNet(model.latent_dim).cuda()
optimizer, lr = get_optimizer(ds=opt.dataset, model=model, architecture=opt.model)
kannet_optimizer = torch.optim.SGD(kannet.parameters(), lr=0.1)
kannet_scheduler = lr_scheduler.CosineAnnealingLR(
    kannet_optimizer, T_max=opt.num_epochs, eta_min=0.0001
)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=opt.num_epochs, eta_min=0.0001
)
# train the network
prv_acc = 0.0
print("Start training ...")
for epoch in range(opt.num_epochs):
    scheduler.step()
    kannet_scheduler.step()
    total_ws = 0.0
    total_robust = 0.0
    total_natural = 0.0
    for i, ((label_x, label_y), (unlab_x, unlab_y)) in enumerate(
        zip(cycle(label_loader), unlab_loader)
    ):
        label_x, label_y = label_x.cuda(), label_y.cuda()
        unlab_x, unlab_y = unlab_x.cuda(), unlab_y.cuda()

        step_natural, step_robust, step_ws = train(
            model, kannet, label_x, label_y, unlab_x, optimizer, kannet_optimizer, epoch
        )

        total_robust += step_robust
        total_natural += step_natural
        total_ws += step_ws
        # print(i)

    total_robust = total_robust / i
    total_natural = total_natural / i
    total_ws = total_ws / i
    test_acc = accuracy(model, test_loader)
    writer.add_scalar("Acc/test", test_acc, epoch)

    log_str = "|Epoch: {:>6}| Loss robust: {:.6f}| WS loss: {:.6f} |Natural loss: {:.6f}| Test Accuracy: {:.6f} |".format(
        epoch, total_robust.item(), total_ws.item(), total_natural.item(), test_acc
    )
    print(log_str)
    write_log(log_str, log_file)

    if test_acc > prv_acc:
        print("Saving ...")
        os.makedirs(os.path.join("ckpt_semi", *prefix), exist_ok=True)
        PATH = os.path.join("ckpt_semi", *(prefix + ["ckpt_{}.pt".format(epoch)]))
        # torch.save(model, PATH)
        prv_acc = test_acc
        write_log("BEST", log_file)
    writer.add_scalar("Loss/loss_robust", total_robust.item(), epoch)
    writer.add_scalar("Loss/loss_natural", total_natural.item(), epoch)


test_acc = accuracy(model, test_loader)
print("Test Accuracy: {}".format(test_acc))
writer.add_scalar("Acc/test", test_acc, epoch)

print("Saving last...")
os.makedirs(os.path.join("ckpt_semi", *prefix), exist_ok=True)
PATH = os.path.join("ckpt_semi", *(prefix + ["ckpt_{}_last.pt".format(epoch)]))
# torch.save(model, PATH)
