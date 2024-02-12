import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape

class Net(nn.Module):
    def __init__(self, config, kernel_size):
        super(Net, self).__init__()
        k1 = kernel_size
        kernel_size = 5
        if config.apply_gauss:
           in_channels = 2*config.mapping_size
        elif config.transform == 'stockwell':
           in_channels = len(config.channels_list)
           h_shape = int(config.fmax*config.sig_time)+1
           w_shape = config.window_len
        elif config.transform == 'csp':
           in_channels = 1
           h_shape = 20
           w_shape = config.window_len
        elif config.transform == 'nothing':
           h_shape = config.n_channels
           w_shape = config.window_len
           in_channels = 1
        elif config.transform == 'wpt':
           h_shape = 20
           w_shape = config.window_len
           in_channels = len(config.channels_list)
        else:
           w_shape = config.n_channels
           h_shape = config.freq_cut
           in_channels = 1

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels, 60, kernel_size=(k1,kernel_size), stride=(1,kernel_size)),
                           nn.ReLU(inplace=True))

        print(f'H: {h_shape} W:{w_shape}')

        h_shape = ((h_shape-k1)+1)
        w_shape = ((w_shape-kernel_size)//kernel_size+1)

        self.conv_layer2 = nn.Sequential(nn.Conv2d(60, 80, kernel_size=(k1,kernel_size), stride=(1,kernel_size)),
#                           nn.BatchNorm2d(80),
                           nn.ReLU(inplace=True))

        h_shape = ((h_shape-k1) + 1)
        w_shape = ((w_shape-kernel_size)//kernel_size + 1)

        self.conv_layer3 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=(k1,kernel_size), stride=(1,kernel_size)),
#                           nn.BatchNorm2d(80),
                           nn.ReLU(inplace=True),
                           nn.Dropout(0.5),
                           nn.AvgPool2d(2))

        h_shape = ((h_shape-k1) + 1)//2
        w_shape = ((w_shape-kernel_size)//kernel_size + 1)//2


        self.conv_layer4 = nn.Sequential(nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.BatchNorm2d(80),
                           nn.Dropout(0.5),
                           nn.ReLU(inplace=True),nn.AvgPool2d(2))

        h_shape = ((h_shape-k1) + 1)//2
        w_shape = ((w_shape-kernel_size) + 1)//2

        print(f'H: {h_shape} W:{w_shape}')
        self.fc1 = nn.Linear(80 * h_shape * w_shape, 1)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        x   = self.conv_layer1(x)
        x   = self.conv_layer2(x)
        x   = self.conv_layer3(x)
        x   = self.conv_layer4(x)
        x   = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        out = self.fc1(x)
        return out

