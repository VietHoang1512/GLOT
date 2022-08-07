import argparse
import os
import os.path as osp
import random

import network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_list import ImageList
from torchvision import transforms
from tqdm import tqdm
from vat import LVATLoss, VATLoss


def seed_everything(seed: int) -> None:
    """
    Seed for reproceducing.
    Args:
        seed (int): seed number
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(
    args, model, train_loader, train_loader1, optimizer, epoch, start_epoch, method
):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    args.log_interval = num_iter

    loss_value = 0
    loss_target_value = 0
    for batch_idx in tqdm(range(num_iter), total=num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        if epoch > start_epoch:
            if method == "LVAT":
                with torch.no_grad():
                    feature_target = model.encode(data_target).detach()
                vat_loss = LVATLoss(xi=args.xi, eps=args.eps, ip=1)
                transfer_loss = vat_loss(model.classifier, feature_target)
            elif method == "VAT":
                lvat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=1)
                transfer_loss = lvat_loss(model, data_target)
            elif method == "WDR":
                pass
            else:
                transfer_loss = torch.tensor(0.0).detach()
        loss_target_value += transfer_loss.item() / args.log_interval
        _, output_source = model(data_source)
        classifier_loss = nn.CrossEntropyLoss()(output_source, label_source)
        loss = args.alpha * transfer_loss + classifier_loss
        loss.backward()
        optimizer.step()
        loss_value += classifier_loss.item() / args.log_interval

        if batch_idx % args.log_interval == args.log_interval - 1:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * args.batch_size,
                    num_iter * args.batch_size,
                    100.0 * batch_idx / num_iter,
                    loss.item(),
                )
            )
            print(
                "transfer_loss: {:.6f} classifier_loss: {:.6f}".format(
                    loss_target_value, loss_value
                )
            )
            loss_value = 0
            loss_target_value = 0


def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        _, output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.data.cpu().max(1, keepdim=True)[1]
        correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser(description="ALDA SVHN2MNIST")
    parser.add_argument("--method", type=str, choices=["VAT", "LVAT", "WDR", "SA"])
    parser.add_argument("--task", default="SVHN2MNIST", help="task to perform")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=200,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, metavar="LR")
    parser.add_argument("--gpu_id", type=str, default=0, help="cuda device id")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--trade_off", type=float, default=1.0, help="trade_off")
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="begin adaptation after start_epoch"
    )
    parser.add_argument("--xi", default=1e-2, type=float, help="VAT xi")
    parser.add_argument("--eps", default=1, type=float, help="VAT epsilon")
    parser.add_argument("--alpha", default=0.1, type=float, help="VAT trade off")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory of our model (in ../snapshot directory)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="all",
        help="whether add reg_loss or correct_loss.",
    )
    parser.add_argument(
        "--cos_dist",
        type=str2bool,
        default=False,
        help="the classifier uses cosine similarity.",
    )
    parser.add_argument("--num_worker", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    source_list = "./data/svhn2mnist/svhn_balanced.txt"
    target_list = "./data/svhn2mnist/mnist_train.txt"
    test_list = "./data/svhn2mnist/mnist_test.txt"

    source_list = open(source_list).readlines()
    target_list = open(target_list).readlines()
    test_list = open(test_list).readlines()

    train_loader = torch.utils.data.DataLoader(
        ImageList(
            source_list,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
            mode="RGB",
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True,
    )
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(
            target_list,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
            mode="RGB",
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ImageList(
            test_list,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
            mode="RGB",
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )

    model = network.SVHN_EnsembNet()
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    start_epoch = args.start_epoch
    if args.output_dir is None:
        args.output_dir = args.task.lower() + "_" + args.method
    output_path = "snapshot/" + args.output_dir
    if os.path.exists(output_path):
        print("checkpoint dir exists, which will be removed")
        import shutil

        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.3
        train(
            args,
            model,
            train_loader,
            train_loader1,
            optimizer,
            epoch,
            start_epoch,
            args.method,
        )
        test(args, model, test_loader)
        if epoch % 5 == 1:
            torch.save(
                model.state_dict(), osp.join(output_path, "epoch_{}.pth".format(epoch))
            )


if __name__ == "__main__":
    main()
