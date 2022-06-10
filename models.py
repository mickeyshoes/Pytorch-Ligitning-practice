from torch import nn

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