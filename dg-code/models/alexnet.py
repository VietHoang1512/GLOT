import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet

# __all__ = ["AlexNet", "alexnet"]

# model_urls = {
#     "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
# }


# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#         self.latent_dim = 256 * 6 * 6
#     def forward(self, x):
#         end_points = {}

#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         end_points["Embedding"] = x

#         x = self.classifier(x)
#         end_points["Predictions"] = F.softmax(input=x, dim=-1)

#         return x, end_points


# def alexnet(pretrained=True, **kwargs):
#     model = AlexNet(**kwargs)

#     if pretrained:
#         pretrained_dict = model_zoo.load_url(model_urls["alexnet"])

#         model_dict = model.state_dict()
#         # 1. filter out unnecessary keys
#         pretrained_dict = {
#             k: v.data
#             for k, v in pretrained_dict.items()
#             if k in model_dict and v.shape == model_dict[k].size()
#         }
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 3. load the new state dict
#         model.load_state_dict(model_dict)
#     return model

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent_dim = 256 * 6 * 6

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, num_classes)

    def forward(self, x):
        end_points = {}

        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        end_points["Embedding"] = x
        x = F.dropout(x,training = self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training = self.training)
        x = F.relu(self.fc4(x))
        x = self.fc6(x)
        end_points["Predictions"] = F.softmax(input=x, dim=-1)

        return x, end_points

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def alexnet(**kwargs):
    model = AlexNet(**kwargs)

    pretrained_dict = model_zoo.load_url(
        "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth")

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    update_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].size():
            update_dict[k] = v.data
        else:
            k = k.replace("classifier.", "fc")
            if v.shape == model_dict[k].size():
                update_dict[k] = v.data
            else:
                print(k, v.shape, model_dict[k].size(), "not match")

    # 2. overwrite entries in the existing state dict
    model_dict.update(update_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


# if __name__ == "__main__":
#     model = alexnet(num_classes=10)
#     print(model)
