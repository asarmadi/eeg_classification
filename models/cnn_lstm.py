import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMConv(nn.Module):

    def __init__(self, config, kernel_size):
        super(LSTMConv, self).__init__()
        self.num_layers  = config.layer_dim
        self.hidden_size = config.hidden_dim
        self.device      = config.device
        self.lstm        = nn.LSTM(config.n_channels, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.1)
        h_shape          = config.window_len

        channel_size     = config.hidden_dim

        self.conv_layer1 = nn.Sequential(nn.Conv1d(config.hidden_dim, channel_size, kernel_size=kernel_size),nn.BatchNorm1d(channel_size),nn.ReLU(inplace=True), nn.Dropout(0.1))
        h_shape          = (h_shape - kernel_size + 1)

        self.conv_layer2 = nn.Sequential(nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size),nn.ReLU(inplace=True),nn.Dropout(0.1))
        h_shape          = (h_shape-kernel_size + 1)

        self.conv_layer3 = nn.Sequential(nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size),nn.ReLU(inplace=True),nn.Dropout(0.1))
        h_shape          = (h_shape-kernel_size+1)

#        self.fc1 = nn.Linear(channel_size*h_shape, 128)
#        self.bn1 = nn.BatchNorm1d(128)
#        self.dro = nn.Dropout(0.1)
        self.fc2 = nn.Linear(channel_size*h_shape, config.n_classes)
        self.sigmoid = nn.LogSoftmax(dim=1)
#        self.relu    = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1)
#        print(x.shape)
 #       input('enter')
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
#        hn = hn.reshape(-1, self.num_layers*self.hidden_size) #reshaping the data for Dense layer next
 #       print(out.shape, hn.shape, cn.shape)
  #      input('enter')
        out = out.permute(0,2,1)
        out = self.conv_layer1(out)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.reshape(-1, out.shape[1]*out.shape[2])
 #       out = self.dro(self.relu(self.bn1(self.fc1(out))))
        out = self.sigmoid(self.fc2(out))
        return out

