import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import reshape

class Net(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_shape[0], in_shape[0]*2, kernel_size=3, stride=2),
                           nn.BatchNorm2d(in_shape[0]*2),
                           nn.Tanh())
#                           nn.ReLU(inplace=True))
#                           nn.Dropout())
#                           nn.AvgPool2d(2,stride=2))

        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_shape[0]*2, in_shape[0]*2, kernel_size=3, stride=2),
                           nn.BatchNorm2d(in_shape[0]*2),
                           nn.Tanh())
#                           nn.ReLU(inplace=True))
 #                          nn.Dropout())
#                           nn.AvgPool2d(2,stride=2))

        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_shape[0]*2, in_shape[0]*2, kernel_size=3, stride=2),
                           nn.BatchNorm2d(in_shape[0]*2),
                           nn.Tanh())
#                           nn.MaxPool2d(2),
#                           nn.ReLU(inplace=True))
  #                         nn.Dropout())
        self.conv_layer4 = nn.Sequential(nn.Conv2d(in_shape[0]*2, in_shape[0]*2, kernel_size=3, stride=2),
                           nn.BatchNorm2d(in_shape[0]*2),
                           nn.ReLU(inplace=True))
#                           nn.AvgPool2d(2,stride=2))


        self.fc1 = nn.Linear(in_shape[0]*2 * 2 * 5, out_shape)
#        self.fc2 = nn.Linear(64, out_shape)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x   = self.conv_layer1(x)
#        print(x.shape)
        x   = self.conv_layer2(x)
 #       print(x.shape)
        x   = self.conv_layer3(x)
  #      print(x.shape)
        x   = self.conv_layer4(x)
#        print(x.shape)
#        input('enter')
        x   = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
#        print(x.shape)
 #       input('enter')
#        x   = self.dropout(self.relu(self.fc1(x)))
        x   = self.fc1(x)
        out = F.log_softmax(x, dim=1)
        return out

