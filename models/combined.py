import torch
import torch.nn as nn
from torch.autograd import Variable

class Combined(nn.Module):

    def __init__(self, config, kernel_size):
        super(Combined, self).__init__()
        k1  = kernel_size
        self.num_layers  = config.layer_dim
        self.hidden_size = config.hidden_dim
        self.device      = config.device
        if config.apply_gauss:
           in_channels = 2*config.mapping_size
        else:
           in_channels = 1
        channel_size   = config.hidden_dim

        self.lstm      = nn.LSTM(config.n_channels, config.hidden_dim, config.layer_dim, batch_first=True)
        self.lstm_conv = nn.Sequential(nn.Conv1d(config.hidden_dim, channel_size, kernel_size=kernel_size),nn.BatchNorm1d(channel_size),nn.ReLU(inplace=True),\
                                        nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size),nn.ReLU(inplace=True),nn.MaxPool1d(2),\
                                        nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size),nn.ReLU(inplace=True),nn.MaxPool1d(2),
                                        nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size),nn.ReLU(inplace=True),nn.MaxPool1d(2),nn.Flatten())

#        k1, kernel_size = 3, 9
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, 60, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True),\
                           nn.Conv2d(60, 80, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True), nn.MaxPool2d(2),\
                           nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2),
                           nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.BatchNorm2d(80),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2),
                           nn.Conv2d(80, 80, kernel_size=(k1,kernel_size)),
                           nn.BatchNorm2d(80),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2),
                           nn.Flatten())

        self.fc1 = nn.Linear(21760, config.n_classes)
        self.sigmoid = nn.LogSoftmax(dim=1)
#        self.relu    = nn.ReLU()

    def forward(self, x):
        out_conv = self.conv2d(x.unsqueeze(1))
        # Propagate input through LSTM
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x.permute(0,2,1), (h_0, c_0)) #lstm with input, hidden, and internal state
        out_lstm      = self.lstm_conv(out.permute(0,2,1)) #lstm with input, hidden, and internal state

#        print(out_lstm.shape, out_conv.shape)
        out = torch.cat((out_lstm,out_conv), 1)
#        print(out.shape)
#        input('enter')
        out = self.sigmoid(self.fc1(out))
        return out

