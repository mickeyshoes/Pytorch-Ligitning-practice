from torch import nn
from torchvision import models

class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.Linear(32,10)
        )

    def forward(self, x):
        return self.layer_1(x)

class LitResnet(nn.Module):

    def __init__(self, num_classes):
        self.backbone = models.resnet18(pretrained=True)
        self.classifiers = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.classifiers(x)