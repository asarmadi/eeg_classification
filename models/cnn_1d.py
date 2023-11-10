import torch
import torch.nn as nn

class Conv1D(nn.Module):

    def __init__(self, config):
        super(Conv1D, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv1d(config.n_channels, 8*config.n_channels, kernel_size=11),nn.BatchNorm1d(8*config.n_channels),nn.ReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(nn.Conv1d(8*config.n_channels, 8*config.n_channels, kernel_size=11),nn.BatchNorm1d(8*config.n_channels),nn.ReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(nn.Conv1d(8*config.n_channels, 8*config.n_channels, kernel_size=11),nn.BatchNorm1d(8*config.n_channels),nn.ReLU(inplace=True))

        self.avgpool = nn.AvgPool2d(2,stride=2)

        self.fc1 = nn.Linear(8*config.n_channels//2*(config.window_len-30)//2, config.n_classes)
#        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
 #       self.fc2 = nn.Linear(config.hidden_dim, config.n_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_layer1(x.permute(0,2,1))
        out = self.conv_layer2(out)
        out = self.avgpool(self.conv_layer3(out))
#        print(out.shape)
 #       input('enter')
#        out,_ = self.rnn(out)
        out = out.reshape(-1, out.shape[1]*out.shape[2])
#        out = self.relu(self.bn1(self.fc1(out)))
        out = self.sigmoid(self.fc1(out))
        return out

