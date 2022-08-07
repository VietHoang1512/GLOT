import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def SelfTraining_loss(logits, prob, threshold=None):
    """
    :param prob: probability of pred (N, C, H, W)
    :return: loss
    """
    ignore_index = -1
    batch_size = prob.size(0) // 2
    logits = logits[batch_size:]
    prob = prob[batch_size:]

    maxpred = torch.argmax(prob.detach(), dim=1)
    mask = maxpred > threshold
    label = torch.where(
        mask, maxpred, torch.ones(1).to(prob.device, dtype=torch.long) * ignore_index
    )

    loss = F.cross_entropy(logits, label, ignore_index=ignore_index)

    return loss


def DANN(features, ad_net):
    """
    :param features: features extracted by the generator (N, hidden_size, H, W)
    :param ad_net: the discriminator network
    :return: loss
    """
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = (
        torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size))
        .float()
        .cuda()
    )
    return nn.BCELoss()(ad_out, dc_target)


def create_matrix(n):
    """
    :param n: matrix size (class num)
    :return a matrix with torch.tensor type:
    for example n=3:
    1     -1/2  -1/2
    -1/2    1   -1/2
    -1/2  -1/2    1
    """
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                a[i, j] = 1
            else:
                a[i, j] = -1 / (n - 1)
    return torch.from_numpy(a).cuda()


def ALDA_loss(ad_out_score, labels_source, softmax_out, weight_type=1, threshold=0.9):
    """
    :param ad_out_score: the discriminator output (N, C, H, W)
    :param labels_source: the source ground truth (N, H, W)
    :param softmax_out: the model prediction probability (N, C, H, W)
    :return:
    adv_loss: adversarial learning loss
    reg_loss: regularization term for the discriminator
    correct_loss: corrected self-training loss
    """
    ad_out = torch.sigmoid(ad_out_score)

    batch_size = ad_out.size(0) // 2
    class_num = ad_out.size(1)

    labels_source_mask = (
        torch.zeros(batch_size, class_num)
        .to(ad_out.device)
        .scatter_(1, labels_source.unsqueeze(1), 1)
    )
    probs_source = softmax_out[:batch_size].detach()
    probs_target = softmax_out[batch_size:].detach()
    maxpred, argpred = torch.max(probs_source, dim=1)
    preds_source_mask = (
        torch.zeros(batch_size, class_num)
        .to(ad_out.device)
        .scatter_(1, argpred.unsqueeze(1), 1)
    )
    maxpred, argpred = torch.max(probs_target, dim=1)
    preds_target_mask = (
        torch.zeros(batch_size, class_num)
        .to(ad_out.device)
        .scatter_(1, argpred.unsqueeze(1), 1)
    )

    # filter out those low confidence samples
    target_mask = maxpred > threshold
    preds_target_mask = torch.where(
        target_mask.unsqueeze(1), preds_target_mask, torch.zeros(1).to(ad_out.device)
    )
    # construct the confusion matrix from ad_out. See the paper for more details.
    confusion_matrix = create_matrix(class_num)
    ant_eye = (1 - torch.eye(class_num)).cuda().unsqueeze(0)
    confusion_matrix = ant_eye / (class_num - 1) + torch.mul(
        confusion_matrix.unsqueeze(0), ad_out.unsqueeze(1)
    )  # (2*batch_size, class_num, class_num)
    preds_mask = torch.cat(
        [preds_source_mask, preds_target_mask], dim=0
    )  # labels_source_mask
    loss_pred = torch.mul(confusion_matrix, preds_mask.unsqueeze(1)).sum(dim=2)
    # different correction targets for different domains
    loss_target = (1 - preds_target_mask) / (class_num - 1)
    loss_target = torch.cat([labels_source_mask, loss_target], dim=0)
    if not ((loss_pred >= 0).all() and (loss_pred <= 1).all()):
        raise AssertionError
    mask = torch.cat([(maxpred >= 0), target_mask], dim=0)
    adv_loss = nn.BCELoss(reduction="none")(loss_pred, loss_target)[mask]
    adv_loss = torch.sum(adv_loss) / mask.float().sum()

    # reg_loss
    reg_loss = nn.CrossEntropyLoss()(ad_out_score[:batch_size], labels_source)

    # corrected target loss function
    target_probs = 1.0 * softmax_out[batch_size:]
    correct_target = torch.mul(
        confusion_matrix.detach()[batch_size:], preds_target_mask.unsqueeze(1)
    ).sum(dim=2)
    correct_loss = -torch.mul(target_probs, correct_target)
    correct_loss = torch.mean(correct_loss[target_mask])
    return adv_loss, reg_loss, correct_loss


def EntropicWassersteinLoss(
    z_s: torch.Tensor,
    ot_cost: torch.Tensor,
    kantorovich_pot: nn.Module,
    epsilon: float = 0.1,
):
    """
    Computes the stochastic entropic optimal transport cost moving points z_s to z_t.
    """
    source_size = ot_cost.size(0)
    target_size = ot_cost.size(1)
    kantorovich_val = kantorovich_pot(z_s)

    exp_term = (-ot_cost + kantorovich_val.repeat(1, target_size)) / epsilon
    kan_network_loss = kantorovich_val.mean()
    ot_loss = (
        torch.mean(
            -epsilon
            * (
                torch.log(torch.tensor(1.0 / source_size))
                + torch.logsumexp(exp_term, dim=0)
            )
        )
        + kan_network_loss
    )

    return ot_loss


def pairwise_cosine(a, b, eps=1e-8):
    """
    Computes the pairwise cosine distance between two tensors.
    Args:
        a: source tensor
        b: target tensor
        eps: epsilon for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return 1 - sim_mt


def pairwise_forward_kl(y_s: torch.Tensor, y_t: torch.Tensor, from_logits=True):
    """
    Computes the pairwise forward KL divergence between two distributions.
    Args:
        y_s: source distribution
        y_t: target distribution
        from_logits: whether y_s and y_t are logits
    """
    if from_logits:
        y_s_prob = F.softmax(y_s, dim=1)
        y_t_prob = F.softmax(y_t, dim=1)
        return (y_t_prob * y_t_prob.log()).sum(dim=1) - torch.einsum(
            "ik, jk -> ij", y_s_prob.log(), y_t_prob
        )
    return (y_t * y_t.log()).sum(dim=1) - torch.einsum("ik, jk -> ij", y_s.log(), y_t)


def pairwise_reverse_kl(y_s: torch.Tensor, y_t: torch.Tensor, from_logits=True):
    """
    Computes the pairwise reverse KL divergence between two distributions.
    Args:
        y_s: source distribution
        y_t: target distribution
        from_logits: whether y_s and y_t are logits
    """
    if from_logits:
        y_s_prob = F.softmax(y_s, dim=1)
        y_t_prob = F.softmax(y_t, dim=1)
        return (
            (y_s_prob * y_s_prob.log()).sum(dim=1)
            - torch.einsum("ik, jk -> ij", y_t_prob.log(), y_s_prob)
        ).t()
    return (
        (y_s * y_s.log()).sum(dim=1) - torch.einsum("ik, jk -> ij", y_t.log(), y_s)
    ).t()


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def MMDLoss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)
    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2
