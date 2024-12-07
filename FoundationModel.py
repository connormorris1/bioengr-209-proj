from torch import nn
from torchvision import models
import torch

# Note almost all this code is taken from RadImageNet's PyTorch example
# https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb
class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        # x = torch.softmax(x, dim=-1)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)

def initialize_radimagenet_resnet(file_path, num_classes, freeze_encoder):
    backbone = Backbone()
    classifier = Classifier(num_class=num_classes)
    model = nn.Sequential(backbone, classifier)
    backbone.load_state_dict(torch.load(file_path))

    # Freeze all parameters in encoder if tag is selected
    if freeze_encoder:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    return model