import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        h_shape = int(config.fmax*config.sig_time)+1
        w_shape = config.window_len

        print(f"Channel: {len(config.channels_list)}, H: {h_shape}, W: {w_shape}")
        self.fn1   = nn.Linear(len(config.channels_list)*h_shape*w_shape, 60)
        self.bn1   = nn.BatchNorm1d(60)

        self.fn2   = nn.Linear(60, 30)
        self.bn2   = nn.BatchNorm1d(30)

        self.fn3   = nn.Linear(30, 1)

        self.relu     = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x   = self.flatten(x)
        x   = self.relu(self.bn1(self.fn1(x)))
        x   = self.relu(self.bn2(self.fn2(x)))
        x   = self.sigmoid(self.fn3(x))
        return x

