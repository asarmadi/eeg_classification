import torch
import torch.nn as nn

class LSTMConv(nn.Module):

    def __init__(self, config, kernel_size):
        super(LSTMConv, self).__init__()
        self.rnn = nn.LSTM(config.window_len, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.1)
        h_shape = config.hidden_dim

        self.conv_layer1 = nn.Sequential(nn.Conv1d(config.n_channels, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))
        h_shape = (h_shape - kernel_size + 1)//2

        self.conv_layer2 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))
        h_shape = (h_shape-kernel_size + 1)//2

        self.conv_layer3 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))
        h_shape = (h_shape-kernel_size+1)//2

        self.avgpool = nn.AvgPool2d(2,stride=2)

        self.fc1 = nn.Linear(4*h_shape, config.n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out,_ = self.rnn(x)
        out = self.conv_layer1(out)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.reshape(-1, out.shape[1]*out.shape[2])
        out = self.sigmoid(self.fc1(out))
        return out

