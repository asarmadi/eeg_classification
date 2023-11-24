import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape

class Net(nn.Module):
    def __init__(self, config, kernel_size):
        super(Net, self).__init__()
        k1 = 5
        self.conv_layer1 = nn.Sequential(nn.Conv2d(2*config.mapping_size, 20, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True))
#        h_shape = config.freq_cut
#        w_shape = config.nTimeBins

        h_shape = config.window_len
        w_shape = config.n_channels
        h_shape = (h_shape-k1)+1
        w_shape = (w_shape-kernel_size)+1
#        print(f'H: {h_shape} W:{w_shape}')

        self.conv_layer2 = nn.Sequential(nn.Conv2d(20, 40, kernel_size=(k1,kernel_size), stride=2),
                           nn.ReLU(inplace=True))
        h_shape = (h_shape-k1)//2 + 1
        w_shape = (w_shape-kernel_size)//2 + 1
#        print(f'H: {h_shape} W:{w_shape}')

        self.conv_layer3 = nn.Sequential(nn.Conv2d(40, 40, kernel_size=(k1,kernel_size), stride=2),
                           nn.ReLU(inplace=True))
        h_shape = (h_shape-k1)//2 + 1
        w_shape = (w_shape-kernel_size)//2 + 1
#        print(f'H: {h_shape} W:{w_shape}')

        self.conv_layer4 = nn.Sequential(nn.Conv2d(40, 40, kernel_size=(k1,kernel_size), stride=2),
                           nn.BatchNorm2d(40),
                           nn.ReLU(inplace=True))
        h_shape = (h_shape-k1)//2 + 1
        w_shape = (w_shape-kernel_size)//2 + 1
#        print(f'H: {h_shape} W:{w_shape}')



        self.fc1 = nn.Linear(40 * h_shape * w_shape, 40)
        self.fc2 = nn.Linear(40, config.n_classes)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x   = self.conv_layer1(x)
        x   = self.conv_layer2(x)
        x   = self.conv_layer3(x)
        x   = self.conv_layer4(x)
        x   = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
#        print(x.shape)
 #       input('enter')
        x     = self.relu(self.fc1(x))
        out   = self.sigmoid(self.fc2(x))
#        out = F.log_softmax(x, dim=1)
        return out

