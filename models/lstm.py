import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMClassifier(nn.Module):

    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.layer_dim  = config.layer_dim
        self.device     = config.device
        self.rnn1       = nn.LSTM(config.n_channels, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.1)
        self.rnn2       = nn.LSTM(config.hidden_dim, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.1)
        self.fc1        = nn.Linear(config.window_len*config.hidden_dim, 64)
        self.fc2        = nn.Linear(64, config.n_classes)
        self.tanh       = nn.Tanh()
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
#        h0, c0 = self.init_hidden(x)
#        out, (hn, cn) = self.rnn(x, (h0, c0))
#        out = self.fc(out[:, -1, :])
        x   = x.permute(0,2,1)
        h_0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.rnn1(x, (h_0, c_0)) #lstm with input, hidden, and internal state
#        out,_ = self.rnn1(x)
#        out,_ = self.rnn2(out)
        out   = out.reshape(-1, out.shape[1]*out.shape[2])
        out   = self.tanh(self.fc1(out))
        out   = self.fc2(out)
        return out
