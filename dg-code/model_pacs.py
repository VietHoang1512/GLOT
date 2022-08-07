import os
import yaml
import numpy as np
from scipy.misc import imresize
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

from common.utils import *
from models.alexnet import alexnet


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


class BatchImageGenerator:
    def __init__(self, flags, stage, data_paths, b_unfold_label):

        if stage not in ["train", "test"]:
            assert ValueError("invalid stage!")

        self.configuration(flags, stage, data_paths)
        for i, path in enumerate(data_paths):
            if i == 0:
                self.images, self.labels = self.load_data(path, b_unfold_label)
            else:
                images, labels = self.load_data(path, b_unfold_label)
                self.images = np.append(self.images, images, axis=0)
                self.labels = np.append(self.labels, labels, axis=0)
        self.file_num_train = len(self.labels)
        print("Total num loaded:", self.file_num_train)

    def configuration(self, flags, stage, data_paths):
        self.batch_size = flags.batch_size
        self.current_index = -1
        self.data_paths = data_paths
        self.stage = stage

    def normalize(self, inputs):

        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = []
        for item in inputs:
            item = np.transpose(item, (2, 0, 1))
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)

            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        inputs_norm = np.stack(inputs_norm)

        return inputs_norm

    def load_data(self, data_path, b_unfold_label):
        def img_resize(x):
            x = x[
                :, :, [2, 1, 0]
            ]  # we use the pre-read hdf5 data file from the download page and need to change BRG to RGB
            return imresize(x, (224, 224, 3))

        if self.stage is "train":
            f = h5py.File(data_path + "_train.hdf5", "r")
            images = np.array(f["images"])
            labels = np.array(f["labels"])
            f.close()
            # f = h5py.File(data_path + "_val.hdf5", "r")
            # images = np.append(images, np.array(f['images']), 0)
            # labels = np.append(labels, np.array(f['labels']), 0)
            # f.close()
        else:
            f = h5py.File(data_path + "_test.hdf5", "r")
            images = np.array(f["images"])
            labels = np.array(f["labels"])
            f.close()

        # N = 1000
        # images = images[:N]
        # labels = labels[:N]

        # resize the image to 224 for the pretrained model
        images = np.array(list(map(img_resize, images)))

        # norm the image value
        images = self.normalize(images)

        assert np.max(images) < 5.0 and np.min(images) > -5.0

        # shift the labels to start from 0
        labels -= np.min(labels)

        if b_unfold_label:
            labels = unfold_label(labels=labels, classes=len(np.unique(labels)))
        assert len(images) == len(labels)

        print("Loaded", len(labels), "samples from", data_path)

        if self.stage is "train":
            images, labels = shuffle_data(samples=images, labels=labels)
        return images, labels

    def get_images_labels_batch(self):

        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.images, self.labels = shuffle_data(
                    samples=self.images, labels=self.labels
                )

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels

    def shuffle(self):
        self.file_num_train = len(self.labels)
        self.current_index = 0
        self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)


class ModelBaseline:
    def __init__(self, flags):

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.network = alexnet(num_classes=flags.num_classes)
        self.network = self.network.cuda()

        print(self.network)
        print("flags:", flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, "flags_log.txt")
        write_log(flags, flags_log)
        with open(os.path.join(flags.logs, "config.yaml"), "w") as outfile:
            yaml.dump(vars(flags), outfile, default_flow_style=False)
        self.load_state_dict(flags, self.network)

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = ["art_painting", "cartoon", "photo", "sketch"]
        del train_data[flags.unseen_index]

        test_data = ["art_painting", "cartoon", "photo", "sketch"]

        self.train_data_paths = []
        for data in train_data:
            data_path = os.path.join(root_folder, data)
            self.train_data_paths.append(data_path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, "path_log.txt")
        write_log(str(self.train_data_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrain = BatchImageGenerator(
            flags=flags,
            data_paths=self.train_data_paths,
            stage="train",
            b_unfold_label=False,
        )

        self.batImageGenTest = BatchImageGenerator(
            flags=flags,
            data_paths=[self.unseen_data_path],
            stage="test",
            b_unfold_label=False,
        )

    def load_state_dict(self, flags, nn):

        if flags.state_dict:

            try:
                tmp = torch.load(flags.state_dict)
                if "state" in tmp.keys():
                    pretrained_dict = tmp["state"]
                else:
                    pretrained_dict = tmp
            except:
                pretrained_dict = model_zoo.load_url(flags.state_dict)

            model_dict = nn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }

            print(
                "model dict keys:",
                len(model_dict.keys()),
                "pretrained keys:",
                len(pretrained_dict.keys()),
            )
            print(
                "model dict keys:",
                model_dict.keys(),
                "pretrained keys:",
                pretrained_dict.keys(),
            )
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            nn.load_state_dict(model_dict)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = sgd(
            parameters=self.network.parameters(),
            lr=flags.lr,
            weight_decay=flags.weight_decay,
            momentum=flags.momentum,
        )

        self.scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1
        )
        # self.scheduler = lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, flags.step_size
        # )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, flags):
        self.network.train()
        # self.network.bn_eval()
        self.best_accuracy = -1

        for ite in range(flags.loops_train):

            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            total_loss = 0.0

            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)
            ), torch.from_numpy(np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = (
                Variable(inputs, requires_grad=False).cuda(),
                Variable(labels, requires_grad=False).long().cuda(),
            )
            # print(inputs)
            # forward with the adapted parameters
            outputs, _ = self.network(x=inputs)

            # loss
            loss = self.loss_fn(outputs, labels)

            total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    "ite:",
                    ite,
                    "total loss:",
                    total_loss.cpu().item(),
                    "lr:",
                    self.scheduler.get_lr()[0],
                )

            flags_log = os.path.join(flags.logs, "loss_log.txt")
            write_log(str(total_loss.item()), flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(flags, ite)

    def test_workflow(self, flags, ite):

        acc_test = self.test(flags=flags, batImageGenTest=self.batImageGenTest)
        self.best_accuracy = max(self.best_accuracy, acc_test)

        f = open(os.path.join(flags.logs, "best_test.txt"), mode="a")
        f.write(
            "ite:{}, test accuracy:{} {}\n".format(
                ite, acc_test, "best!" if acc_test == self.best_accuracy else ""
            )
        )
        f.close()

        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)

        outfile = os.path.join(flags.model_path, "best_model.tar")
        torch.save({"ite": ite, "state": self.network.state_dict()}, outfile)

    def test(self, flags, batImageGenTest):

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
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(
                    torch.from_numpy(np.array(test_image_split, dtype=np.float32))
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

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print("----------accuracy test----------:", accuracy)

        # switch on the network train mode
        self.network.train()
        return accuracy


class ModelWDR(ModelBaseline):
    def __init__(self, flags):
        super(ModelWDR, self).__init__(flags)

    def maximize(self, flags):
        self.network.eval()

        images_train, labels_train = (
            self.batImageGenTrain.images,
            self.batImageGenTrain.labels,
        )
        images, labels = [], []

        for start, end in tqdm(
            zip(
                range(0, len(labels_train), flags.batch_size),
                range(flags.batch_size, len(labels_train), flags.batch_size),
            ),
            total=len(
                list(
                    zip(
                        range(0, len(labels_train), flags.batch_size),
                        range(flags.batch_size, len(labels_train), flags.batch_size),
                    )
                )
            ),
            desc="Generating adversarial samples",
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
            random_noise = flags.xi * torch.randn(x_adv.shape).to(inputs.device)

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
                    torch.max(x_adv, x_particle - flags.xi), x_particle + flags.xi
                )
                x_adv = torch.clamp(x_adv, -2.64, 2.64).detach()
                self.network.zero_grad()
            images.append(x_adv.cpu().numpy())
            labels.append(y.cpu().numpy())

        self.network.train()
        return np.concatenate(images), np.concatenate(labels)

    def train(self, flags):
        self.network.train()
        # self.network.bn_eval()
        self.best_accuracy = -1
        counter_k = 0
        for ite in range(flags.loops_train):
            if ((ite + 1) % flags.loops_min == 0) and (
                counter_k < flags.k
            ):  # if T_min iterations are passed
                print("Generating adversarial images [iter {}]".format(counter_k))
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
            total_loss = 0.0

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

            total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    "ite:",
                    ite,
                    "total loss:",
                    total_loss.cpu().item(),
                    "lr:",
                    self.scheduler.get_lr()[0],
                )

            flags_log = os.path.join(flags.logs, "loss_log.txt")
            write_log(str(total_loss.item()), flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(flags, ite)
