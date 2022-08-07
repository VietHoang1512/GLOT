import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class KanNet(nn.Module):
    """Kantorovich potential neural network"""

    def __init__(self, input_dim=2048, bottleneck_dim=512):
        super(KanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, 1)
        self.fc = nn.Linear(input_dim, 1)
        # self.fc1.apply(init_weights)
        # self.fc2.apply(init_weights)
        # self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.fc(x)
        return x

    def get_parameters(self):
        parameter_list = [
            {"params": self.fc.parameters(), "lr_mult": 1, "decay_mult": 2},
            {"params": self.fc1.parameters(), "lr_mult": 1, "decay_mult": 2},
            {"params": self.fc2.parameters(), "lr_mult": 1, "decay_mult": 2},
        ]
        return parameter_list
    
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
