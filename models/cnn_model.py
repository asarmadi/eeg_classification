import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape

class Net(nn.Module):
    def __init__(self, config, kernel_size):
        super(Net, self).__init__()
        k1 = kernel_size
        self.conv_layer1 = nn.Sequential(nn.Conv2d(1, 60, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True))
#        h_shape = config.freq_cut
 #       w_shape = config.nTimeBins

        h_shape = config.window_len
        w_shape = config.n_channels
        h_shape = (h_shape-k1)+1
        w_shape = (w_shape-kernel_size)+1
#        print(f'H: {h_shape} W:{w_shape}')

        self.conv_layer2 = nn.Sequential(nn.Conv2d(60, 80, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True))
#                           nn.AvgPool2d(2))

        h_shape = ((h_shape-k1) + 1)
        w_shape = ((w_shape-kernel_size) + 1)
#        print(f'H: {h_shape} W:{w_shape}')

        self.conv_layer3 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True),
                           nn.Dropout(0.2), nn.AvgPool2d(2))

        h_shape = ((h_shape-k1) + 1)//2
        w_shape = ((w_shape-kernel_size) + 1)//2

        self.conv_layer4 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.BatchNorm2d(80),
                           nn.ReLU(inplace=True),
                           nn.Dropout(0.2),nn.AvgPool2d(2))

        h_shape = ((h_shape-k1) + 1)//2
        w_shape = ((w_shape-kernel_size) + 1)//2

        print(f'H: {h_shape} W:{w_shape}')
        self.fc1 = nn.Linear(80 * h_shape * w_shape, config.n_classes)
#        self.fc2 = nn.Linear(40, config.n_classes)
        self.relu    = nn.ReLU(inplace=True)
        self.lsf     = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x   = self.conv_layer1(x.unsqueeze(1))
        x   = self.conv_layer2(x)
        x   = self.conv_layer3(x)
        x   = self.conv_layer4(x)
        x   = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        out = self.lsf(self.fc1(x))
        return out

