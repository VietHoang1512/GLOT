import torch

from utils.pgd import pgd_attack


def accuracy(model, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in dataset:
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            predicted = torch.argmax(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total
    # print(correct, total)
    model.train(True)
    return acc


def accuracy_robust(model, dataset, attack_params):
    model.eval()
    correct = 0
    total = 0

    # attack_params = {"loss_type": "ce", "epsilon": 8/255, "step_size": 2/255, "num_steps": 0, "order": "linf", "random_init": True, "x_min": 0.0, "x_max": 1.0}

    for (x, y) in dataset:
        x = x.cuda()
        y = y.cuda()

        x_adv, _ = pgd_attack(model, x, y, x.device, attack_params, status="eval")

        outputs = model(x_adv)
        predicted = torch.argmax(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = correct / total
    # print(correct, total)
    model.train(True)
    return acc
