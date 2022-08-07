import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

NO_LABEL = -1


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("test.png")


def idx_select_dataset(dataset, num_val, num_label=None, num_class=10):
    if num_label is not None:
        l_idxes = []
        ul_idxes = []
        count = {}
        num_label = num_label // num_class
        for idx, (x, y) in enumerate(dataset):
            if y not in count:
                l_idxes.append(idx)
                count[y] = 1
            else:
                if count[y] < num_label:
                    l_idxes.append(idx)
                    count[y] += 1
                else:
                    ul_idxes.append(idx)

        np.random.shuffle(ul_idxes)
        val_idxes = ul_idxes[0:num_val]
        ul_idxes = ul_idxes[num_val:]
        return l_idxes, ul_idxes, val_idxes
    else:
        idxes = []
        for idx, (x, y) in enumerate(dataset):
            idxes.append(idx)

        np.random.shuffle(idxes)
        val_idxes = idxes[0:num_val]
        idxes = idxes[num_val:]
        return idxes, val_idxes


def encode_label(label):
    return NO_LABEL * (label + 1)


def decode_label(label):
    return NO_LABEL * label - 1


def split_relabel_data(np_labs, labels, label_per_class, num_classes):
    """Return the labeled indexes and unlabeled_indexes"""
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs == id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs


def get_data_loader(ds, batch_size=128, num_label=4000):
    if ds == "cifar10":
        channel_stats = dict(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        train_transform = transforms.Compose(
            [
                transforms.Pad(2, padding_mode="reflect"),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**channel_stats),
            ]
        )
        eval_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(**channel_stats)]
        )
        trainset = torchvision.datasets.CIFAR10(
            "./data", train=True, download=True, transform=train_transform
        )
        evalset = torchvision.datasets.CIFAR10(
            "./data", train=False, download=True, transform=eval_transform
        )

    elif ds == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor()])
        eval_transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=train_transform
        )
        evalset = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=eval_transform
        )

    num_classes = 10
    label_per_class = num_label // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
        np.array(trainset.targets), trainset.targets, label_per_class, num_classes
    )
    # dconfig =  {'trainset': trainset, 'evalset': evalset, 'label_idxs': labeled_idxs, 'unlab_idxs': unlabed_idxs, 'num_classes': num_classes}
    ## supervised batch loader
    label_sampler = SubsetRandomSampler(labeled_idxs)
    label_batch_sampler = BatchSampler(label_sampler, batch_size, drop_last=True)
    label_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=label_batch_sampler, num_workers=4
    )
    ## unsupervised batch loader
    unlabed_idxs += labeled_idxs
    unlab_sampler = SubsetRandomSampler(unlabed_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, batch_size, drop_last=True)
    unlab_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=unlab_batch_sampler, num_workers=4
    )
    ## eval batch loader
    test_loader = torch.utils.data.DataLoader(
        evalset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )
    return label_loader, unlab_loader, test_loader
