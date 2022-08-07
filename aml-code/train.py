import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
from functools import partial

from trades import trades_loss
from mart import mart_loss
from pgd import pgd_loss, pgd_attack
from pgd_svgd import pgd_svgd_symkl, pgd_svgd_ce, pgd_svgd_kl
from compact import compact_loss

def train(model, data_loader, epoch, optimizer, device, log_interval, writer): 
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        num_batches = len(data_loader)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        nat_acc = get_acc(output, target)

        if batch_idx % log_interval == 0:    
            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('loss={:.4f}', loss.item()), 
                ('nat_acc={:.4f}', nat_acc.item()), 

            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)

    return writer

def baseline_train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()

    if attack_params['defense'] == 'pgd_train': 
        defense = pgd_loss 
    elif attack_params['defense'] == 'trades_train': 
        defense = trades_loss
    elif attack_params['defense'] == 'mart_train': 
        defense = mart_loss
    else:
        raise ValueError 

    num_batches = len(data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        loss, X_adv = defense(model=model,
                           x_natural=data,
                           y=target,
                           device=device,
                           optimizer=optimizer,
                           step_size=attack_params['step_size'],
                           epsilon=attack_params['epsilon'],
                           perturb_steps=attack_params['num_steps'],
                           alpha=attack_params['alpha'],
                           beta=attack_params['trades_beta'], 
                           projecting=attack_params['projecting'], 
                           x_max=attack_params['x_max'], 
                           x_min=attack_params['x_min'])

        model.train()
        loss.backward()
        optimizer.step()

        nat_output = model(data)
        adv_output = model(X_adv)
        nat_acc = get_acc(nat_output, target)
        adv_acc = get_acc(adv_output, target)

        if batch_idx % log_interval == 0:

            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('nat_acc={:.4f}', nat_acc.item()), 
                ('adv_acc={:.4f}', adv_acc.item()), 
                ('loss={:.4f}', loss.item()), 
            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('adv_acc', adv_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)
    return writer

def svgd_train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()

    if attack_params['defense'] == 'pgd_svgd_symkl': 
        defense = pgd_svgd_symkl 
    elif attack_params['defense'] == 'pgd_svgd_ce':
        defense = pgd_svgd_ce
    elif attack_params['defense'] == 'pgd_svgd_kl':
        defense = pgd_svgd_kl
    else:
        raise ValueError 

    num_batches = len(data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        x_adv, x_natural = defense(model=model,
                           x_natural=data,
                           y=target,
                           device=device,
                           optimizer=optimizer,
                           step_size=attack_params['step_size'],
                           epsilon=attack_params['epsilon'],
                           perturb_steps=attack_params['num_steps'],
                           alpha=attack_params['alpha'],
                           beta=attack_params['trades_beta'], 
                           projecting=attack_params['projecting'], 
                           x_max=attack_params['x_max'], 
                           x_min=attack_params['x_min'], 
                           num_particles=attack_params['num_particles'], 
                           sigma=attack_params['sigma'])

        model.train()

        # 
        y_rp = target.unsqueeze(-1).repeat(attack_params['num_particles'], 1).squeeze(-1)
        nat_output, z_nat = model(x_natural, return_z=True)
        adv_output, z_adv = model(x_adv, return_z=True)

        l_symkl = nn.KLDivLoss(size_average=True)(
                        F.log_softmax(adv_output, dim=1),
                        F.softmax(nat_output, dim=1),
                    ) + nn.KLDivLoss(size_average=True)(
                        F.log_softmax(nat_output, dim=1),
                        F.softmax(adv_output, dim=1),
                    )
        
        l_adv = nn.CrossEntropyLoss(size_average=True)(adv_output, y_rp)
        l_nat = nn.CrossEntropyLoss(size_average=True)(nat_output, y_rp)

        l_com = compact_loss(latents=torch.cat([z_adv, z_nat], dim=0), 
                                    labels= torch.cat([y_rp, y_rp], dim=0), 
                                    num_classes=10, 
                                    dist=attack_params['dist']) 

        loss = attack_params['wkl'] * l_symkl +\
             attack_params['trades_beta'] * l_adv +\
             attack_params['alpha'] * l_nat +\
             attack_params['wcom'] * l_com
        loss.backward()
        optimizer.step()

        nat_output = model(data)
        adv_output = model(x_adv[:data.shape[0]])
        nat_acc = get_acc(nat_output, target)
        adv_acc = get_acc(adv_output, target)

        if batch_idx % log_interval == 0:

            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('nat_acc={:.4f}', nat_acc.item()), 
                ('adv_acc={:.4f}', adv_acc.item()), 
                ('loss={:.4f}', loss.item()), 
                ('l_symkl={:.4f}', l_symkl.item()), 
                ('l_adv={:.4f}', l_adv.item()), 
                ('l_nat={:.4f}', l_nat.item()), 
                ('l_com={:.4f}', l_com.item()), 
            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('adv_acc', adv_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)
    return writer

def test(model, data_loader, device): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    return accuracy

def adv_test(model, data_loader, device, attack_params): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            X_adv, _ = pgd_attack(model, data, target, device, attack_params, status='eval')
            X_adv = Variable(X_adv.data, requires_grad=False)

            output = model(X_adv)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nRobustness evaluation : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    return accuracy

def gen_pgd_adv(model, data_loader, device, attack_params): 
    model.eval()
    all_adv = []
    all_target = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            X_adv, _ = pgd_attack(model, data, target, device, attack_params, status='eval')
            X_adv = Variable(X_adv.data, requires_grad=False)
            all_adv.append(X_adv)
            all_target.append(target)
    
    all_adv = torch.cat(all_adv, dim=0)
    all_target = torch.cat(all_target, dim=0)

    return all_adv, all_target

def get_pred(model, data_loader, device): 
    model.eval()
    result = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            output = torch.nn.Softmax()(output)
            result.append(output.cpu().numpy())

    result = np.concatenate(result, axis=0)
    return result

def get_acc(output, target): 
    pred = output.argmax(dim=1, keepdim=True)
    acc = torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor))
    return acc 
