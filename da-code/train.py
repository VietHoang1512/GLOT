import argparse
import os
import os.path as osp
import random

import lr_schedule
import network
import numpy as np
import pre_process as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from data_list import ImageList, ImageList_label
from loss import (
    EntropicWassersteinLoss,
    pairwise_cosine,
    pairwise_forward_kl,
    pairwise_reverse_kl,
)
from potential_net import KanNet
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from vat import VATLoss
from wdr import gen_adv


def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    dataset = loader["test"]
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(dataset[i]) for i in range(10)]
            for i in range(len(dataset[0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    feature, predict_out = model(inputs[j])
                    predict_out = nn.Softmax(dim=1)(predict_out)
                    outputs.append(predict_out)
                outputs = sum(outputs) / 10
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(dataset)
            for i in range(len(dataset)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feature, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    return accuracy


def image_label(loader, model, threshold=0.9, out_dir=None):
    # save the pseudo_label
    out_path = osp.join(out_dir, "pseudo_label.txt")
    print("Pseudo Labeling to ", out_path)
    iter_label = iter(loader["target_label"])
    with torch.no_grad():
        with open(out_path, "w") as f:
            for i in range(len(loader["target_label"])):
                inputs, labels, paths = iter_label.next()
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                softmax_outputs = nn.Softmax(dim=1)(outputs)
                maxpred, pseudo_labels = torch.max(softmax_outputs, dim=1)
                pseudo_labels[maxpred < threshold] = -1
                for (path, label) in zip(paths, pseudo_labels):
                    f.write(path + " " + str(label.item()) + "\n")
    return out_path


def train(config):
    # set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]["params"])
    prep_dict["target"] = prep.image_train(**config["prep"]["params"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]["params"])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]["params"])

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    source_list = open(data_config["source"]["list_path"]).readlines()
    target_list = open(data_config["target"]["list_path"]).readlines()

    dsets["source"] = ImageList(source_list, transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(
        dsets["source"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=config["args"].num_worker,
        drop_last=True,
    )
    dsets["target"] = ImageList(target_list, transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=config["args"].num_worker,
        drop_last=True,
    )
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = open(data_config["test"]["list_path"]).readlines()
            dsets["test"] = [
                ImageList(test_list, transform=prep_dict["test"][i]) for i in range(10)
            ]
            dset_loaders["test"] = [
                DataLoader(
                    dset,
                    batch_size=test_bs,
                    shuffle=False,
                    num_workers=config["args"].num_worker,
                )
                for dset in dsets["test"]
            ]
    else:
        test_list = open(data_config["test"]["list_path"]).readlines()
        dsets["test"] = ImageList(test_list, transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(
            dsets["test"],
            batch_size=test_bs,
            shuffle=False,
            num_workers=1,
        )

    dsets["target_label"] = ImageList_label(target_list, transform=prep_dict["target"])
    dset_loaders["target_label"] = DataLoader(
        dsets["target_label"],
        batch_size=test_bs,
        shuffle=False,
        num_workers=config["args"].num_worker,
        drop_last=False,
    )

    # set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    kannet = KanNet(input_dim=net_config["params"]["bottleneck_dim"]).cuda()
    model = nn.Sequential(base_network.bottleneck, base_network.classifier)

    if config["restore_path"]:
        checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))[
            "base_network"
        ]
        ckp = {}
        for k, v in checkpoint.items():
            if "module" in k:
                ckp[k.split("module.")[-1]] = v
            else:
                ckp[k] = v
        base_network.load_state_dict(ckp)
        log_str = "successfully restore from {}".format(
            osp.join(config["restore_path"], "best_model.pth")
        )
        config["out_file"].write(log_str + "\n")
        config["out_file"].flush()
        print(log_str)

    # add additional network for some methods
    parameter_list = base_network.get_parameters()
    kannet_parameter_list = kannet.get_parameters()

    # set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](
        parameter_list, **(optimizer_config["optim_params"])
    )
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    kannet_optimizer_config = config["optimizer"]
    kannet_optimizer = kannet_optimizer_config["type"](
        kannet_parameter_list, **(optimizer_config["optim_params"])
    )
    kannet_param_lr = []
    for param_group in kannet_optimizer.param_groups:
        kannet_param_lr.append(param_group["lr"])
    kannet_schedule_param = optimizer_config["lr_param"]

    if config["args"].kl == "forward":
        pairwise_kl = pairwise_forward_kl
    elif config["args"].kl == "reverse":
        pairwise_kl = pairwise_reverse_kl
    else:
        raise ValueError("kl must be forward or reverse")

    if config["args"].distance == "euclidean":
        pairwise_distance = torch.cdist
    elif config["args"].distance == "cosine":
        pairwise_distance = pairwise_cosine
    else:
        raise ValueError("distance must be euclidean or cosine")

    writer = SummaryWriter(config["output_path"])

    # train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    loss_value = 0
    source_lds_value = 0
    target_lds_value = 0
    constrastive_value = 0
    ws_distance_value = 0
    pbar = tqdm(range(config["num_iterations"]), total=config["num_iterations"])
    for i in pbar:
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(
                dset_loaders, base_network, test_10crop=prep_config["test_10crop"]
            )
            temp_model = base_network  # nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_step = i
                best_acc = temp_acc
                best_model = temp_model
                checkpoint = {"base_network": best_model.state_dict()}
                # torch.save(
                #     checkpoint, osp.join(config["output_path"], "best_model.pth")
                # )
                print("\n##########     save the best model.    #############\n")
            log_str = "\niter: {:05d}, precision: {:.5f}".format(i, temp_acc) + (
                " best!" if best_acc == temp_acc else ""
            )
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            writer.add_scalar("precision", temp_acc, i)
            print(log_str)

            print(
                f"Classification loss: {loss_value:.4f}\tconstrastive loss: {constrastive_value:.4f}\tWasserstein distance: {ws_distance_value:.4f}"
            )
            print(
                f"Local distribution smoothness:\tSource: {source_lds_value:.4f}\ttarget: {target_lds_value:.4f}"
            )

            loss_value = 0
            constrastive_value = 0
            source_lds_value = 0
            target_lds_value = 0
            ws_distance_value = 0
            # show val result on tensorboard
            # images_inv = prep.inv_preprocess(inputs_source.cpu(), 3)
            # for index, img in enumerate(images_inv):
            #     writer.add_image(str(index) + "/Images", img, i)

        if i > config["stop_step"]:
            log_str = "method {}, iter: {:05d}, precision: {:.5f}".format(
                config["output_path"], best_step, best_acc
            )
            config["final_log"].write(log_str + "\n")
            config["final_log"].flush()
            break

        # train one iter
        base_network.train()
        kannet.train()
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        kannet_optimizer = lr_scheduler(kannet_optimizer, i, **kannet_schedule_param)
        kannet_optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = (
            Variable(inputs_source).cuda(),
            Variable(inputs_target).cuda(),
            Variable(labels_source).cuda(),
        )

        target_lds = 0.0
        constrastive_loss = torch.tensor(0.0).detach()
        source_lds = torch.tensor(0.0).detach()
        target_lds = torch.tensor(0.0).detach()

        # align source-target
        with torch.no_grad():
            z_source = base_network.get_embedding(inputs_source).detach()
            z_target = base_network.get_embedding(inputs_target).detach()
            y_source = base_network.classifier(z_source).detach()
            y_target = base_network.classifier(z_target).detach()
        cost_matrix = pairwise_distance(z_source, z_target) + config[
            "args"
        ].label_shift * pairwise_kl(y_source, y_target)

        loss_ws_st = -EntropicWassersteinLoss(
            z_source, cost_matrix, kannet, config["args"].epsilon
        )
        loss_ws_st.backward()
        kannet_optimizer.step()
        kannet.zero_grad()

        # TODO: check the VAT loss function
        if config["method"] == "LVAT":
            with torch.no_grad():
                feature_target = base_network.encode(inputs_target).detach()
            vat_loss = VATLoss(
                xi=config["args"].xi, eps=config["args"].eps, ip=config["args"].niter
            )
            target_lds = vat_loss(
                nn.Sequential(base_network.bottleneck, base_network.classifier),
                feature_target,
            )
        elif config["method"] == "VAT":
            vat_loss = VATLoss(
                xi=config["args"].xi, eps=config["args"].eps, ip=config["args"].niter
            )
            target_lds = vat_loss(base_network, inputs_target)
        elif config["method"] == "WDR":
            with torch.no_grad():
                feature_source, outputs_source = base_network(inputs_source)
                feature_source = feature_source.detach().repeat(
                    config["args"].n_particles, 1
                )
                outputs_source = outputs_source.detach().repeat(
                    config["args"].n_particles, 1
                )

                feature_target, outputs_target = base_network(inputs_target)
                feature_target = feature_target.detach().repeat(
                    config["args"].n_particles, 1
                )
                outputs_target = outputs_target.detach().repeat(
                    config["args"].n_particles, 1
                )

            feature_source += config["args"].xi * torch.randn(feature_source.shape).to(feature_source.device)

            feature_target += config["args"].xi * torch.randn(feature_target.shape).to(feature_target.device)

            feature_target_adv = gen_adv(
                model,
                feature_target,
                outputs_target,
                config["args"].n_particles,
                None,
                config["args"].eps,
                config["args"].niter,
                config["target_adv"],
            )

            feature_source_adv = gen_adv(
                model,
                feature_source,
                outputs_source,
                config["args"].n_particles,
                None,
                config["args"].eps,
                config["args"].niter,
                config["source_adv"],
            )

            base_network.zero_grad()

            outputs_target_adv = model(feature_target_adv)
            embeddings = base_network.bottleneck(feature_source_adv)
            outputs_source_adv = base_network.classifier(embeddings)

            source_lds = F.kl_div(
                F.log_softmax(outputs_source_adv, dim=1),
                F.softmax(outputs_source, dim=1),
                reduction="mean",
            ) + F.kl_div(
                F.log_softmax(outputs_source, dim=1),
                F.softmax(outputs_source_adv, dim=1),
                reduction="mean",
            )

            target_lds = F.kl_div(
                F.log_softmax(outputs_target_adv, dim=1),
                F.softmax(outputs_target, dim=1),
                reduction="mean",
            ) + F.kl_div(
                F.log_softmax(outputs_target, dim=1),
                F.softmax(outputs_target_adv, dim=1),
                reduction="mean",
            )

            distance = LpDistance(normalize_embeddings=False)
            # mining_func = miners.TripletMarginMiner(
            #     margin=config["args"].margin, distance=distance, type_of_triplets="semihard"
            # )
            labels = labels_source.repeat(config["args"].n_particles)
            # indices_tuple = mining_func(embeddings, labels)
            constrastive_loss = losses.TripletMarginLoss(
                margin=config["args"].margin,
                distance=distance,
            )(embeddings, labels)

        z_source = base_network.get_embedding(inputs_source[:train_bs])
        z_target = base_network.get_embedding(inputs_target[:train_bs])
        outputs_source = base_network.classifier(z_source)
        outputs_target = base_network.classifier(z_target)
        cost_matrix = pairwise_distance(z_source, z_target) + config[
            "args"
        ].label_shift * pairwise_kl(outputs_source, outputs_target)

        loss_ws_st = EntropicWassersteinLoss(
            z_source, cost_matrix, kannet, config["args"].epsilon
        )
        classifier_loss = nn.CrossEntropyLoss()(
            outputs_source[:train_bs], labels_source
        )
        loss_value += classifier_loss.item() / config["test_interval"]
        source_lds_value += source_lds.item() / config["test_interval"]
        target_lds_value += target_lds.item() / config["test_interval"]
        constrastive_value += constrastive_loss.item() / config["test_interval"]
        ws_distance_value += loss_ws_st.item() / config["test_interval"]
        total_loss = (
            classifier_loss
            + config["args"].source * source_lds
            + config["args"].target * target_lds
            + config["args"].beta * constrastive_loss
            + config["args"].align * loss_ws_st
        )
        total_loss.backward()
        optimizer.step()
        pbar.set_postfix(
            {
                "Classification": classifier_loss.item(),
                "Source": source_lds.item(),
                "Target": target_lds.item(),
                "Constrastive": constrastive_loss.item(),
                "Align": loss_ws_st.item(),
            }
        )
    checkpoint = {"base_network": temp_model.state_dict()}
    # torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    return best_acc


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser(
        description="Conditional Domain Adversarial Network"
    )
    parser.add_argument("--method", type=str, choices=["VAT", "LVAT", "WDR", "SO"])

    parser.add_argument(
        "--net",
        type=str,
        default="ResNet50",
        choices=[
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "VGG11",
            "VGG13",
            "VGG16",
            "VGG19",
            "VGG11BN",
            "VGG13BN",
            "VGG16BN",
            "VGG19BN",
            "AlexNet",
        ],
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="office",
        choices=["office", "image-clef", "visda", "office-home"],
        help="The dataset or source dataset used",
    )
    parser.add_argument(
        "--s_dset_path",
        type=str,
        default="data/office-31/amazon.txt",
        help="The source dataset path list",
    )
    parser.add_argument(
        "--t_dset_path",
        type=str,
        default="data/office-31/webcam.txt",
        help="The target dataset path list",
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=250,
        help="interval of two continuous test phase",
    )
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=5000,
        help="interval of two continuous output model",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="test",
        help="experiment name",
    )
    parser.add_argument(
        "--restore_dir",
        type=str,
        default=None,
        help="restore directory of our model (in ../snapshot directory)",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--source", type=float, default=0.0, help="source lds trade off"
    )
    parser.add_argument(
        "--target", type=float, default=1.0, help="target lds trade off"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="constrastive loss trade off"
    )
    parser.add_argument(
        "--margin", type=float, default=0.3, help="constrastive loss margin"
    )
    parser.add_argument("--xi", default=1e-2, type=float, help="VAT xi")
    parser.add_argument("--eps", default=1, type=float, help="VAT epsilon")
    parser.add_argument(
        "--n_particles", default=2, type=int, help="number of adversarial particles"
    )
    parser.add_argument(
        "--niter", default=1, type=int, help="number of SVGD iterations"
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Latent distance metric",
    )
    parser.add_argument(
        "--kl",
        type=str,
        default="forward",
        choices=["forward", "reverse", "symmetric"],
        help="Label distance metric",
    )
    parser.add_argument(
        "--align", type=float, default=0.1, help="source target align trade off"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="sinkhorn entropy regularization"
    )
    parser.add_argument("--label_shift", type=float, default=0.5, help="label shift")
    parser.add_argument(
        "--batch_size", type=int, default=36, help="training batch size"
    )
    parser.add_argument(
        "--label_interval",
        type=int,
        default=200,
        help="interval of two continuous pseudo label phase",
    )
    parser.add_argument("--stop_step", type=int, default=0, help="stop steps")
    parser.add_argument("--final_log", type=str, default=None, help="final_log file")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--test_10crop", type=str2bool, default=True)

    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # train config
    config = {}
    config["args"] = args
    config["method"] = args.method
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = os.path.join(
        "outputs/",
        args.dset,
        os.path.basename(args.s_dset_path).split(".")[0]
        + "2"
        + os.path.basename(args.t_dset_path).split(".")[0],
        args.method,
        args.exp,
    )
    config["restore_path"] = (
        "snapshot/" + args.restore_dir if args.restore_dir else None
    )
    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil

        shutil.rmtree(config["output_path"], ignore_errors=True)
    os.makedirs(config["output_path"], exist_ok=True)

    config["prep"] = {
        "test_10crop": args.test_10crop,
        "params": {"resize_size": 256, "crop_size": 224},
    }
    config["loss"] = {
        "source": args.source,
        "target": args.target,
        "xi": args.xi,
        "eps": args.eps,
    }
    if "ResNet" in args.net:
        net = network.ResNetFc
        config["network"] = {
            "name": net,
            "params": {"resnet_name": args.net, "bottleneck_dim": 512},
        }
    elif "VGG" in args.net:
        config["network"] = {
            "name": network.VGGFc,
            "params": {"vgg_name": args.net, "bottleneck_dim": 256},
        }

    config["optimizer"] = {
        "type": optim.SGD,
        "optim_params": {
            "lr": args.lr,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "nesterov": True,
        },
        "lr_type": "inv",
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
    }

    config["dataset"] = args.dset
    config["data"] = {
        "source": {"list_path": args.s_dset_path, "batch_size": args.batch_size},
        "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size},
        "test": {"list_path": args.t_dset_path, "batch_size": 4},
    }

    if config["dataset"] == "office":
        if (
            ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path)
            or ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path)
            or ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path)
        ):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or (
            "dslr" in args.s_dset_path and "webcam" in args.t_dset_path
        ):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
            args.stop_step = 20000
        else:
            config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
        args.stop_step = 20000
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError("Dataset has not been implemented.")
    if args.lr != 0.001:
        config["optimizer"]["lr_param"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["gamma"] = 0.001

    if args.stop_step == 0:
        config["stop_step"] = 20000
    else:
        config["stop_step"] = args.stop_step

    with open(osp.join(config["output_path"], "config.yaml"), "w") as outfile:
        print(config)
        yaml.dump(config, outfile, default_flow_style=False)
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    config["out_file"].write(str(config))
    config["out_file"].flush()
    config["source_adv"] = open(osp.join(config["output_path"], "source_adv.txt"), "w")
    config["target_adv"] = open(osp.join(config["output_path"], "target_adv.txt"), "w")
    if args.final_log is None:
        config["final_log"] = open("log.txt", "a")
    else:
        config["final_log"] = open(args.final_log, "a")

    train(config)
