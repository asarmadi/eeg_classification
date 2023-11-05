import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(config.n_channels,   config.n_channels*4, kernel_size=9, stride=2),
                           nn.ReLU(inplace=True))

        self.conv_layer2 = nn.Sequential(nn.Conv2d(config.n_channels*4, config.n_channels*4, kernel_size=9, stride=2),
                           nn.ReLU(inplace=True))

        self.conv_layer3 = nn.Sequential(nn.Conv2d(config.n_channels*4, config.n_channels*4, kernel_size=5, stride=2),
                           nn.ReLU(inplace=True))

        self.conv_layer4 = nn.Sequential(nn.Conv2d(config.n_channels*4, config.n_channels*4, kernel_size=5),
                           nn.BatchNorm2d(config.n_channels*4),
                           nn.ReLU(inplace=True),
                           nn.AvgPool2d(2,stride=2))


        self.fc1 = nn.Linear(config.n_channels*4 * 2 * 8, 40)
        self.fc2 = nn.Linear(40, config.n_classes)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x   = self.conv_layer1(x)
#        print(x.shape)
        x   = self.conv_layer2(x)
#        print(x.shape)
        x   = self.conv_layer3(x)
#        print(x.shape)
        x   = self.conv_layer4(x)
#        print(x.shape)
 #       input('enter')
        x   = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
#        print(x.shape)
 #       input('enter')
        x     = self.relu(self.fc1(x))
        out   = self.sigmoid(self.fc2(x))
#        out = F.log_softmax(x, dim=1)
        return out

