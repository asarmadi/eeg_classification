import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fn1   = nn.Linear(120, 60)
        self.bn1   = nn.BatchNorm1d(60)

        self.fn2   = nn.Linear(60, 30)
        self.bn2   = nn.BatchNorm1d(30)

        self.fn3   = nn.Linear(30, config.n_classes)

        self.relu     = nn.ReLU()
        self.sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x   = self.relu(self.bn1(self.fn1(x)))
        x   = self.relu(self.bn2(self.fn2(x)))
        x   = self.sigmoid(self.fn3(x))
        return x

