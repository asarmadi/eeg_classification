import torch
import torch.nn as nn

class LSTMConv(nn.Module):

    def __init__(self, config):
        super(LSTMConv, self).__init__()
        kernel_size = 5
        self.conv_layer1 = nn.Sequential(nn.Conv1d(config.hidden_dim, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=kernel_size),nn.AvgPool2d(2),nn.ReLU(inplace=True))

        self.rnn = nn.LSTM(config.n_channels, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.1)

        self.avgpool = nn.AvgPool2d(2,stride=2)

        self.fc1 = nn.Linear(4*234, config.n_classes)
#        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
 #       self.fc2 = nn.Linear(config.hidden_dim, config.n_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
#        out = x.permute(0,2,1)
#        print(out.shape)
        out,_ = self.rnn(x)
        out = out.permute(0,2,1)
        out = self.conv_layer1(out)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
#        print(out.shape)
 #       input('enter')
        out = out.reshape(-1, out.shape[1]*out.shape[2])
#        out = self.relu(self.bn1(self.fc1(out)))
        out = self.sigmoid(self.fc1(out))
        return out

