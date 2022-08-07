from tqdm import tqdm
import contextlib
import os
import pickle
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import yaml
from models.lenet import LeNet5
from common.data_gen_MNIST import BatchImageGenerator, get_data_loaders
from common.utils import (
    fix_all_seed,
    write_log,
    adam,
    sgd,
    compute_accuracy,
    entropy_loss,
)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


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


class ModelBaseline(object):
    def __init__(self, flags):

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:",
              torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.network = LeNet5(num_classes=flags.num_classes)
        self.network = self.network.cuda()

        print(self.network)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        with open(os.path.join(flags.logs, "config.yaml"), "w") as outfile:
            yaml.dump(vars(flags), outfile, default_flow_style=False)

    def setup_path(self, flags):

        root_folder = "data"
        data, data_loaders = get_data_loaders()

        seen_index = flags.seen_index
        self.train_data = data[seen_index]
        self.test_data = [x for index, x in enumerate(
            data) if index != seen_index]

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, "path_log.txt")
        write_log(str(self.train_data), flags_log)
        write_log(str(self.test_data), flags_log)

        self.batImageGenTrain = BatchImageGenerator(
            flags=flags,
            stage="train",
            file_path=root_folder,
            data_loader=data_loaders[seen_index],
            b_unfold_label=False,
        )

        self.batImageGenTests = []
        for index, test_loader in enumerate(data_loaders):
            if index != seen_index:
                batImageGenTest = BatchImageGenerator(
                    flags=flags,
                    stage="test",
                    file_path=root_folder,
                    data_loader=test_loader,
                    b_unfold_label=False,
                )
                self.batImageGenTests.append(batImageGenTest)
        self.best_accuracies_test = [-1] * len(self.batImageGenTests)

    def configure(self, flags):

        for name, param in self.network.named_parameters():
            print(name, param.size())

        self.optimizer = adam(
            parameters=self.network.parameters(),
            lr=flags.lr,
            weight_decay=flags.weight_decay,
        )

        self.scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=flags.step_size, gamma=0.3
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, flags):
        self.network.train()
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):

            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader

            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(labels, requires_grad=False).long().cuda(),
            )

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
            str_log = f"|iter: {ite} |total loss: {loss.cpu().item()} |lr: {self.scheduler.get_last_lr()[0]}|"
            if ite < 500 or ite % 500 == 0:
                print(str_log)

            flags_log = os.path.join(flags.logs, "loss_log.txt")
            write_log(str_log, flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenTests, flags, ite)

    def test_workflow(self, batImageGenTests, flags, ite):

        accuracies = []
        for count, batImageGenTest in enumerate(batImageGenTests):
            accuracy_test = self.test(
                batImageGenTest=batImageGenTest,
                ite=ite,
                log_dir=flags.logs,
                log_prefix="test_index_{}".format(count),
                best_accuracy_test=self.best_accuracies_test[count],
            )
            self.best_accuracies_test[count] = max(
                self.best_accuracies_test[count], accuracy_test
            )
            accuracies.append(accuracy_test)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_test:
            self.best_accuracy_test = mean_acc

            f = open(os.path.join(flags.logs, "best_test.txt"), mode="a")
            str_log = "ite:{}, best test accuracy:{}\n".format(
                ite, self.best_accuracy_test
            )
            f.write(str_log)
            f.close()
            print(str_log)
            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            # outfile = os.path.join(flags.model_path, "best_model.tar")
            # torch.save(
            #     {"ite": ite, "state": self.network.state_dict()}, outfile)

    def test(
        self, batImageGenTest, ite, log_prefix, log_dir="logs/", best_accuracy_test=-1
    ):

        # switch on the network test mode
        self.network.eval()

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(
                    int(len(images_test) * (per_slice + 1) / n_slices_test)
                )
            test_image_splits = np.split(
                images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(
                    torch.from_numpy(
                        np.array(test_image_split, dtype=np.float32))
                ).cuda()
                tuples = self.network(images_test)

                predictions = tuples[-1]["Predictions"]
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(
                torch.from_numpy(np.array(images_test, dtype=np.float32))
            ).cuda()
            tuples = self.network(images_test)

            predictions = tuples[-1]["Predictions"]
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(
            predictions=predictions, labels=labels_test)
        print(
            "----------accuracy test----------:",
            accuracy,
            " best!" if (accuracy > best_accuracy_test) else "",
        )

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, "{}.txt".format(log_prefix)), mode="a")
        f.write(
            "ite:{}, accuracy:{}{}\n".format(
                ite, accuracy, " best!" if (
                    accuracy > best_accuracy_test) else ""
            )
        )
        f.close()

        # switch on the network train mode
        self.network.train()

        return accuracy


class ModelADA(ModelBaseline):
    def __init__(self, flags):
        super(ModelADA, self).__init__(flags)

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()

    def maximize(self, flags):
        self.network.eval()

        images_train, labels_train = (
            self.batImageGenTrain.images,
            self.batImageGenTrain.labels,
        )
        images, labels = [], []

        for start, end in zip(
            range(0, len(labels_train), flags.batch_size),
            range(flags.batch_size, len(labels_train), flags.batch_size),
        ):
            inputs, targets = torch.from_numpy(
                np.array(images_train[start:end], dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train[start:end], dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, targets = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(targets, requires_grad=False).long().cuda(),
            )

            inputs_embedding = self.network(
                x=inputs)[-1]["Embedding"].detach().clone()
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
                write_log(
                    "|ite_adv:{}| loss: {}".format(
                        ite_max, loss.item()), flags_log
                )

            inputs_max = inputs_max.detach().clone().clamp(min=0.0, max=1.0)
            images.append(inputs_max.cpu().numpy())
            labels.append(targets.cpu().numpy())

        return np.concatenate(images), np.concatenate(labels)

    def train(self, flags):
        counter_k = 0
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):
            if ((ite + 1) % flags.loops_min == 0) and (
                counter_k < flags.k
            ):  # if T_min iterations are passed
                print(
                    "Generating adversarial images [iter {}]".format(counter_k))
                images, labels = self.maximize(flags)
                self.batImageGenTrain.images = np.concatenate(
                    (self.batImageGenTrain.images, images)
                )
                self.batImageGenTrain.labels = np.concatenate(
                    (self.batImageGenTrain.labels, labels)
                )
                self.batImageGenTrain.shuffle()
                counter_k += 1

            self.network.train()
            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(labels, requires_grad=False).long().cuda(),
            )

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
            # if iter==flags.
            if ite < 500 or ite % 500 == 0:
                print(
                    "ite:",
                    ite,
                    "total loss:",
                    loss.cpu().item(),
                    "lr:",
                    self.scheduler.get_last_lr()[0],
                )

            flags_log = os.path.join(flags.logs, "loss_log.txt")
            write_log(str(loss.item()), flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenTests, flags, ite)


class ModelMEADA(ModelADA):
    def __init__(self, flags):
        super(ModelMEADA, self).__init__(flags)

    def maximize(self, flags):
        self.network.eval()

        images_train, labels_train = (
            self.batImageGenTrain.images,
            self.batImageGenTrain.labels,
        )
        images, labels = [], []
        flags_log = os.path.join(flags.logs, "max_loss_log.txt")
        write_log(f"_" * 30, flags_log)
        for start, end in zip(
            range(0, len(labels_train), flags.batch_size),
            range(flags.batch_size, len(labels_train), flags.batch_size),
        ):
            inputs, targets = torch.from_numpy(
                np.array(images_train[start:end], dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train[start:end], dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, targets = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(targets, requires_grad=False).long().cuda(),
            )

            inputs_embedding = self.network(
                x=inputs)[-1]["Embedding"].detach().clone()
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

                write_log(f"|ite_adv:{ite_max:>3}| {loss.item()}|", flags_log)

            inputs_max = inputs_max.detach().clone().clamp(min=0.0, max=1.0)
            images.append(inputs_max.cpu().numpy())
            labels.append(targets.cpu().numpy())

        return np.concatenate(images), np.concatenate(labels)


class ModelInput(ModelADA):
    def __init__(self, flags):
        super(ModelInput, self).__init__(flags)

    def vat(self, feature_adv, y, flags):
        raise NotImplementedError("Not implemented yet")
        feature_adv = feature_adv.detach().clone()

        with torch.no_grad():
            pred = self.network.decode(feature_adv)
            pred = F.softmax(pred[0] if isinstance(
                pred, tuple) else pred, dim=1)

        # prepare random unit tensor
        d = torch.rand(feature_adv.shape).to(feature_adv.device)
        d = l2_normalize(d)
        flags_log = os.path.join(flags.logs, "vat.txt")
        write_log(f"_" * 30, flags_log)
        with _disable_tracking_bn_stats(self.network):
            # calc adversarial direction
            for ite in range(flags.loops_adv):
                d.requires_grad_()
                pred_hat = self.network.decode(feature_adv + flags.xi * d)
                pred_hat = pred_hat[0] if isinstance(
                    pred_hat, tuple) else pred_hat
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = l2_normalize(d.grad)
                self.network.zero_grad()
                write_log(
                    f"|ite_adv:{ite:>3}| loss {adv_distance.item():.6f}|", flags_log
                )
            # calc LDS
            r_adv = d * flags.eps
            pred_hat = self.network.decode(feature_adv + r_adv)
            pred_hat = pred_hat[0] if isinstance(pred_hat, tuple) else pred_hat
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds

    def maximize(self, flags):
        self.network.eval()
        cache_dir = os.path.join(flags.logs, "cache")
        if os.path.exists(cache_dir):
            print("checkpoint dir exists, which will be removed")
        shutil.rmtree(cache_dir, ignore_errors=True)
        images_train, labels_train = (
            self.batImageGenTrain.images,
            self.batImageGenTrain.labels,
        )
        images, labels = [], []

        for start, end in tqdm(
            zip(
                range(0, len(labels_train), flags.batch_size),
                range(flags.batch_size, len(labels_train), flags.batch_size),
            ), total=len(list(zip(
                range(0, len(labels_train), flags.batch_size),
                range(flags.batch_size, len(labels_train), flags.batch_size),
            ))), desc="Generating adversarial samples"
        ):
            inputs, targets = torch.from_numpy(
                np.array(images_train[start:end], dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train[start:end], dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, targets = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(targets, requires_grad=False).long().cuda(),
            )
            flags_log = os.path.join(flags.logs, "wdr.txt")
            write_log(f"_" * 30, flags_log)
            y = targets.repeat(flags.n_particles)

            kernel = RBF(flags.n_particles)
            batch_size, c, w, h = inputs.size()
            # self.network.eval()
            x_particle = inputs.repeat(flags.n_particles, 1, 1, 1)
            with torch.no_grad():
                y_pred = self.network(x=x_particle)[0]
            x_adv = Variable(x_particle.data, requires_grad=True)
            random_noise = flags.xi * \
                torch.randn(x_adv.shape).to(inputs.device)

            x_adv = Variable(
                (x_adv.data + random_noise).detach().clone(), requires_grad=True
            )
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
                    torch.max(x_adv, x_particle -
                              flags.xi), x_particle + flags.xi
                )
                x_adv = torch.clamp(x_adv, 0, 1).detach()
                self.network.zero_grad()
            # if len(labels) >= 50000:
            #     N = len(os.listdir(cache_dir))
            #     with open(os.path.join(cache_dir, f"{N}.pkl"), "wb") as f:
            #         pickle.dump({"images": images, "labels": labels}, f)
            #     images, labels = [], []
            images.append(x_adv.cpu().numpy())
            labels.append(y.cpu().numpy())

        self.network.train()
        return np.concatenate(images), np.concatenate(labels)


class ModelLatent(ModelBaseline):
    def __init__(self, flags):
        super(ModelLatent, self).__init__(flags)

    def vat(self, feature_adv, labels, flags):
        feature_adv = feature_adv.detach().clone()
        labels = F.one_hot(labels, num_classes=10).float()
        # prepare random unit tensor
        d = torch.rand(feature_adv.shape).to(feature_adv.device)
        d = l2_normalize(d)
        flags_log = os.path.join(flags.logs, "vat.txt")
        write_log(f"_" * 30, flags_log)
        with _disable_tracking_bn_stats(self.network):
            # calc adversarial direction
            for ite in range(flags.loops_adv):
                d.requires_grad_()
                pred_hat = self.network.decode(feature_adv + flags.xi * d)
                pred_hat = pred_hat[0] if isinstance(
                    pred_hat, tuple) else pred_hat
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(
                    logp_hat, labels, reduction="batchmean")
                adv_distance.backward()
                d = l2_normalize(d.grad)
                self.network.zero_grad()
                write_log(
                    f"|ite_adv:{ite:>3}| loss {adv_distance.item():.6f}|", flags_log
                )
            # calc LDS
            r_adv = d * flags.eps
            pred_hat = self.network.decode(feature_adv + r_adv)
            pred_hat = pred_hat[0] if isinstance(pred_hat, tuple) else pred_hat
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, labels, reduction="batchmean")

        return lds

    def wdr(self, feature, outputs, labels, flags):

        labels = labels.repeat(flags.n_particles)
        logits = outputs.detach().clone().repeat(flags.n_particles, 1)
        feature = feature.detach().clone().repeat(flags.n_particles, 1)

        feature_adv = feature + torch.FloatTensor(*feature.shape).uniform_(
            -flags.xi, flags.xi
        ).to(feature.device)

        kernel = RBF(flags.n_particles)
        batch_size = feature_adv.size(0)
        flags_log = os.path.join(flags.logs, f"lwdr_{flags.loss_type}.txt")
        write_log(f"_" * 30, flags_log)
        for ite_max in range(flags.loops_adv):
            feature_adv.requires_grad_()
            with torch.enable_grad():
                outputs_adv = self.network.decode(feature_adv)
                if flags.loss_type == "pgd":
                    loss_kl = nn.CrossEntropyLoss(size_average=False)(
                        outputs_adv, labels
                    )
                elif flags.loss_type == "trade":
                    loss_kl = nn.KLDivLoss(size_average=False)(
                        F.log_softmax(outputs_adv, dim=1),
                        F.softmax(logits, dim=1),
                    )  # + nn.KLDivLoss(size_average=False)(
                    #     F.log_softmax(logits, dim=1),
                    #     F.softmax(outputs_adv, dim=1),
                    # )
                else:
                    raise ValueError(f"{flags.loss_type} is not supported")
                write_log(
                    f"|ite_adv:{ite_max:>3}| loss {loss_kl.item():.6f}|", flags_log
                )
                score_func = torch.autograd.grad(loss_kl, [feature_adv])[0]

                score_func = score_func.reshape(batch_size, -1)

                K_XX = kernel(feature_adv, feature_adv.detach())
                grad_K = -torch.autograd.grad(K_XX.sum(), feature_adv)[0]

                phi = (K_XX.detach().matmul(score_func) + grad_K) / batch_size
                # phi = l2_normalize(phi)
                feature_adv = (feature_adv + flags.eps * phi).detach()

                feature_adv = torch.min(
                    torch.max(feature_adv, feature -
                              flags.xi), feature + flags.xi
                )
                self.network.zero_grad()
        self.network.train()
        with torch.enable_grad():
            outputs_adv = self.network.decode(feature_adv)

            if flags.loss_type == "pgd":
                lds = F.cross_entropy(outputs_adv, labels, reduction="mean")
            else:
                lds = nn.KLDivLoss()(
                    F.log_softmax(outputs_adv, dim=1),
                    F.softmax(logits, dim=1),
                )  # + nn.KLDivLoss()(
                #     F.log_softmax(logits, dim=1),
                #     F.softmax(outputs_adv, dim=1),
                # )

        return lds

    def train(self, flags):
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):
            self.network.train()
            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(labels, requires_grad=False).long().cuda(),
            )

            # forward with the adapted parameters
            with torch.no_grad():
                outputs, end_points = self.network(x=inputs)
            feature = end_points["Feature"]
            # print(feature)
            self.network.zero_grad()
            lds = torch.tensor(0.0)
            if ite > flags.loops_min:
                if flags.algorithm == "LWDR":
                    lds = self.wdr(feature, outputs, labels, flags)
                else:
                    lds = self.vat(feature, labels, flags)
            outputs, _ = self.network(x=inputs)
            classification_loss = self.loss_fn(outputs, labels)
            loss = classification_loss
            if ite > flags.loops_min:
                loss += flags.gamma * lds
            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            loss.backward()

            # optimize the parameters
            self.optimizer.step()
            str_log = f"|ite: {ite:>5} |total loss: {loss.cpu().item():.6f} |classification: {classification_loss.cpu().item():.6f} |local distribution robustness {lds.cpu().item():.6f} |lr: {self.scheduler.get_last_lr()[0]}|"

            if ite < 500 or (ite % 500) <= 50:
                print(str_log)

            flags_log = os.path.join(flags.logs, "loss_log.txt")
            write_log(str_log, flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenTests, flags, ite)
