import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):

    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.layer_dim  = config.layer_dim
        self.rnn1    = nn.LSTM(config.n_channels, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.01)
        self.rnn2    = nn.LSTM(config.hidden_dim, config.hidden_dim, config.layer_dim, batch_first=True, dropout=0.01)
        self.fc1    = nn.Linear(config.n_channels*config.hidden_dim, 64)
        self.fc2    = nn.Linear(64, config.n_classes)
        self.tanh   = nn.ReLU()
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
#        h0, c0 = self.init_hidden(x)
#        out, (hn, cn) = self.rnn(x, (h0, c0))
#        out = self.fc(out[:, -1, :])
#        out   = x.permute(0,2,1)
        print(x.shape)
        out,_ = self.rnn1(x)
        print(out.shape)
        out,_ = self.rnn2(out)
        print(out.shape)
        out   = out.reshape(-1, out.shape[1]*out.shape[2])
        out   = self.tanh(self.fc1(out))
        out   = self.sigmoid(self.fc2(out))
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]
