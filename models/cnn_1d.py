import torch
import torch.nn as nn

class Conv1D(nn.Module):

    def __init__(self, config, kernel_size):
        super(Conv1D, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv1d(config.n_channels, 2*config.n_channels, kernel_size=kernel_size),\
                                         nn.BatchNorm1d(2*config.n_channels),nn.ReLU(inplace=True))
        out_shape = (config.window_len-kernel_size+1)

        self.conv_layer2 = nn.Sequential(nn.Conv1d(2*config.n_channels, 4*config.n_channels, kernel_size=kernel_size),\
                                         nn.BatchNorm1d(4*config.n_channels),nn.ReLU(inplace=True))
        out_shape = (out_shape-kernel_size+1)

        self.conv_layer3 = nn.Sequential(nn.Conv1d(4*config.n_channels, 8*config.n_channels, kernel_size=kernel_size),\
                                         nn.BatchNorm1d(8*config.n_channels),nn.AvgPool1d(2),nn.ReLU(inplace=True))
        out_shape = (out_shape-kernel_size+1)//2

        self.fc1 = nn.Linear(8*config.n_channels*out_shape, config.n_classes)
        self.sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.reshape(-1, out.shape[1]*out.shape[2])
        out = self.sigmoid(self.fc1(out))
        return out

