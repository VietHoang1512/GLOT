import torch.nn as nn


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
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.fc(x)
        return x

    def get_parameters(self):
        parameter_list = [
            {"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2},
            {"params": self.fc1.parameters(), "lr_mult": 10, "decay_mult": 2},
            {"params": self.fc2.parameters(), "lr_mult": 10, "decay_mult": 2},
        ]
        return parameter_list
