from __future__ import print_function, absolute_import, division
import yaml
import os
import torch
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.densenet import densenet
from models.resnext import resnext29
from models.wideresnet import WideResNet
from common.utils import fix_all_seed, write_log, sgd, entropy_loss

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


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
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())


class ModelBaseline(object):
    def __init__(self, flags):

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        if flags.dataset == "cifar10":
            num_classes = 10
        else:
            num_classes = 100

        if flags.model == "densenet":
            self.network = densenet(num_classes=num_classes)
        elif flags.model == "wrn":
            self.network = WideResNet(
                flags.layers, num_classes, flags.widen_factor, flags.droprate
            )
        elif flags.model == "allconv":
            self.network = AllConvNet(num_classes)
        elif flags.model == "resnext":
            self.network = resnext29(num_classes=num_classes)
        else:
            raise Exception("Unknown model.")
        self.network = self.network.cuda()
        self.start_epoch = 0
        if flags.model_ckpt.lower() != "none":
            config = torch.load(flags.model_ckpt)
            self.network.load_state_dict(config["state"])
            print("Loaded model from {}".format(flags.model_ckpt))
            flags.epochs += config["epoch"]
            self.start_epoch = config["epoch"] + 1
        print(self.network)
        print("flags:", flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, "flags_log.txt")
        write_log(flags, flags_log)
        with open(os.path.join(flags.logs, "config.yaml"), "w") as outfile:
            yaml.dump(vars(flags), outfile, default_flow_style=False)

    def setup_path(self, flags):

        root_folder = "data"
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                self.preprocess,
            ]
        )
        self.test_transform = self.preprocess

        if flags.dataset == "cifar10":
            self.train_data = datasets.CIFAR10(
                root_folder, train=True, transform=self.train_transform, download=True
            )
            self.test_data = datasets.CIFAR10(
                root_folder, train=False, transform=self.test_transform, download=True
            )
            self.base_c_path = os.path.join(root_folder, "CIFAR-10-C")
        else:
            self.train_data = datasets.CIFAR100(
                root_folder, train=True, transform=self.train_transform, download=True
            )
            self.test_data = datasets.CIFAR100(
                root_folder, train=False, transform=self.test_transform, download=True
            )
            self.base_c_path = os.path.join(root_folder, "CIFAR-100-C")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=flags.batch_size,
            shuffle=True,
            num_workers=flags.num_workers,
            pin_memory=True,
        )

    def configure(self, flags):

        for name, param in self.network.named_parameters():
            print(name, param.size())

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            flags.lr,
            momentum=flags.momentum,
            weight_decay=flags.weight_decay,
            nesterov=True,
        )

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader) * flags.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, flags):
        self.network.train()
        self.best_accuracy_test = -1

        for epoch in range(0, flags.epochs):
            for i, (images_train, labels_train) in enumerate(self.train_loader):

                # wrap the inputs and labels in Variable
                inputs, labels = images_train.cuda(), labels_train.cuda()

                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)

                # init the grad to zeros first
                self.optimizer.zero_grad()

                # backward your network
                loss.backward()

                # optimize the parameters
                self.optimizer.step()
                self.scheduler.step()

                if epoch < 5 or epoch % 5 == 0:
                    print(
                        "epoch:",
                        epoch,
                        "ite",
                        i,
                        "total loss:",
                        loss.cpu().item(),
                        "lr:",
                        self.scheduler.get_lr()[0],
                    )

                flags_log = os.path.join(flags.logs, "loss_log.txt")
                write_log(str(loss.item()), flags_log)

            self.test_workflow(epoch, flags)

    def test_workflow(self, epoch, flags):
        """Evaluate network on given corrupted dataset."""
        accuracies = []
        for count, corruption in enumerate(CORRUPTIONS):
            # Reference to original data is mutated
            self.test_data.data = np.load(
                os.path.join(self.base_c_path, corruption + ".npy")
            )
            self.test_data.targets = torch.LongTensor(
                np.load(os.path.join(self.base_c_path, "labels.npy"))
            )

            test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=flags.batch_size,
                shuffle=False,
                num_workers=flags.num_workers,
                pin_memory=True,
            )

            accuracy_test = self.test(
                test_loader,
                epoch,
                log_dir=flags.logs,
                log_prefix="test_index_{}".format(count),
            )
            accuracies.append(accuracy_test)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_test:
            self.best_accuracy_test = mean_acc

            f = open(os.path.join(flags.logs, "best_test.txt"), mode="a")
            f.write(
                "epoch:{}, best test accuracy:{}\n".format(
                    epoch, self.best_accuracy_test
                )
            )
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, "best_model.tar")
            torch.save({"epoch": epoch, "state": self.network.state_dict()}, outfile)

    def test(self, test_loader, epoch, log_prefix, log_dir="logs/"):

        # switch on the network test mode
        self.network.eval()

        total_correct = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.cuda(), targets.cuda()
                logits, _ = self.network(images)
                pred = logits.data.max(1)[1]
                total_correct += pred.eq(targets.data).sum().item()

        accuracy = total_correct / len(test_loader.dataset)
        print("----------accuracy test----------:", accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, "{}.txt".format(log_prefix)), mode="a")
        f.write("epoch:{}, accuracy:{}\n".format(epoch, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()

        return accuracy


class ModelADA(ModelBaseline):
    def __init__(self, flags):
        super(ModelADA, self).__init__(flags)

    def setup_path(self, flags):
        super(ModelADA, self).setup_path(flags)
        self.image_denormalise = Denormalise([0.5] * 3, [0.5] * 3)
        self.image_transform = transforms.ToPILImage()

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()

    def maximize(self, flags):
        self.network.eval()

        self.train_data.transform = self.preprocess
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=True,
        )
        images, labels = [], []

        for i, (images_train, labels_train) in enumerate(self.train_loader):

            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()

            # forward with the adapted parameters
            inputs_embedding = self.network(x=inputs)[-1]["Embedding"].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                tuples = self.network(x=inputs_max)

                # loss
                loss = self.loss_fn(tuples[0], targets) - flags.gamma * self.dist_fn(
                    tuples[-1]["Embedding"], inputs_embedding
                )

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                flags_log = os.path.join(flags.logs, "max_loss_log.txt")
                write_log("ite_adv:{}, {}".format(ite_max, loss.item()), flags_log)

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())

        return np.stack(images), labels

    def train(self, flags):
        counter_k = 0
        counter_ite = 0
        self.best_accuracy_test = -1

        for epoch in range(self.start_epoch, flags.epochs):
            # if T_min iterations are passed
            if ((epoch + 1) % flags.epochs_min == 0) and (counter_k < flags.k):
                print("Generating adversarial images [iter {}]".format(counter_k))
                images, labels = self.maximize(flags)
                self.train_data.data = np.concatenate([self.train_data.data, images])
                self.train_data.targets.extend(labels)
                counter_k += 1

            self.network.train()
            self.train_data.transform = self.train_transform
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=True,
            )
            self.scheduler.T_max = counter_ite + len(self.train_loader) * (
                flags.epochs - epoch
            )

            for i, (images_train, labels_train) in enumerate(self.train_loader):
                counter_ite += 1

                # wrap the inputs and labels in Variable
                inputs, labels = images_train.cuda(), labels_train.cuda()

                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)

                # init the grad to zeros first
                self.optimizer.zero_grad()

                # backward your network
                loss.backward()

                # optimize the parameters
                self.optimizer.step()
                self.scheduler.step()

                if epoch < 5 or epoch % 5 == 0:
                    print(
                        "epoch:",
                        epoch,
                        "ite",
                        i,
                        "total loss:",
                        loss.cpu().item(),
                        "lr:",
                        self.scheduler.get_lr()[0],
                    )

                flags_log = os.path.join(flags.logs, "loss_log.txt")
                write_log(str(loss.item()), flags_log)

            self.test_workflow(epoch, flags)
            if epoch + 1 < flags.epochs_min:
                outfile = os.path.join(flags.model_path, "pretrained.pt")
                torch.save(
                    {"epoch": epoch, "state": self.network.state_dict()}, outfile
                )


class ModelMEADA(ModelADA):
    def __init__(self, flags):
        super(ModelMEADA, self).__init__(flags)

    def maximize(self, flags):
        self.network.eval()

        self.train_data.transform = self.preprocess
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=True,
        )
        images, labels = [], []

        for i, (images_train, labels_train) in enumerate(self.train_loader):

            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()

            # forward with the adapted parameters
            inputs_embedding = self.network(x=inputs)[-1]["Embedding"].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                tuples = self.network(x=inputs_max)

                # loss
                loss = (
                    self.loss_fn(tuples[0], targets)
                    + flags.eta * entropy_loss(tuples[0])
                    - flags.gamma
                    * self.dist_fn(tuples[-1]["Embedding"], inputs_embedding)
                )

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                flags_log = os.path.join(flags.logs, "max_loss_log.txt")
                write_log("ite_adv:{}, {}".format(ite_max, loss.item()), flags_log)

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())

        return np.stack(images), labels


class ModelWDR(ModelADA):
    def __init__(self, flags):
        super(ModelWDR, self).__init__(flags)

    def maximize(self, flags):
        self.network.eval()

        self.train_data.transform = self.preprocess
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=32,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=True,
        )
        images, labels = [], []

        for i, (images_train, labels_train) in tqdm(
            enumerate(self.train_loader),
            desc="Generating adversarial images",
            total=len(self.train_loader),
        ):

            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()

            flags_log = os.path.join(flags.logs, "wdr.txt")
            write_log(f"_" * 30, flags_log)
            kernel = RBF(flags.n_particles)
            batch_size, c, w, h = inputs.size()
            with torch.no_grad():
                y_pred = self.network(x=inputs)[0].repeat(flags.n_particles, 1)
            x_particle = inputs.repeat(flags.n_particles, 1, 1, 1)
            x_adv = x_particle.detach().clone()
            y = targets.clone().repeat(flags.n_particles)
            random_noise = flags.xi * torch.randn(x_adv.shape).to(inputs.device)

            x_adv = (x_adv + random_noise).detach().clone()
            for ite_max in range(flags.loops_adv):
                x_adv = x_adv.reshape(batch_size * flags.n_particles, -1)
                x_adv.requires_grad_(True)
                y_adv = self.network(
                    x_adv.reshape(batch_size * flags.n_particles, c, w, h)
                )[0]

                if flags.loss_type == "pgd":
                    loss_kl = nn.CrossEntropyLoss(size_average=False)(y_adv, y)
                elif flags.loss_type == "trade":
                    loss_kl = nn.KLDivLoss(size_average=False)(
                        F.log_softmax(y_adv, dim=1),
                        F.softmax(y_pred, dim=1),
                    ) + nn.KLDivLoss(size_average=False)(
                        F.log_softmax(y_pred, dim=1),
                        F.softmax(y_adv, dim=1),
                    )
                else:
                    raise ValueError(f"{flags.loss_type} is not supported")
                write_log(
                    f"|ite_adv:{ite_max:>3}| loss {loss_kl.item():.6f}|", flags_log
                )

                score_func = torch.autograd.grad(loss_kl, [x_adv])[0]
                K_XX = kernel(x_adv, x_adv.detach())
                grad_K = -torch.autograd.grad(K_XX.sum(), x_adv)[0]

                phi = (K_XX.detach().matmul(score_func) + grad_K) / (
                    batch_size * flags.n_particles
                )
                x_adv = (x_adv + flags.eps * torch.sign(phi)).detach()
                x_adv = x_adv.reshape(batch_size * flags.n_particles, c, w, h)
                x_adv = torch.min(
                    torch.max(x_adv, x_particle - flags.xi), x_particle + flags.xi
                )

            inputs_max = x_adv.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(y[j].item())

        return np.stack(images), labels
