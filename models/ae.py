import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, h_shape, out_dim, in_dim):
        super(Encoder, self).__init__()
        kernel_size   = 21
        self.cnn1     = nn.Conv1d(in_dim, 2*in_dim, kernel_size=kernel_size)
#        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_shape       = ((h_shape-kernel_size)+1)

        self.cnn2     = nn.Conv1d(2*in_dim, 2*in_dim, kernel_size=kernel_size)
#        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_shape       = ((h_shape-kernel_size)+1)

        self.cnn3     = nn.Conv1d(2*in_dim, 2*in_dim, kernel_size=kernel_size)
#        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_shape       = ((h_shape-kernel_size)+1)

        self.relu     = nn.ReLU()
        self.linear   = nn.Linear(2*in_dim*h_shape, out_dim)
        self.flatten  = nn.Flatten()

    def forward(self, x):
        x   = self.relu(self.cnn1(x))
        x   = self.relu(self.cnn2(x))
        x   = self.relu(self.cnn3(x))
        x   = self.flatten(x)
        x   = self.relu(self.linear(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.h_shape, self.in_channels = 415, 1
        kernel_size    = 5
        self.linear    = nn.Linear(in_dim, self.in_channels*self.h_shape)
        self.cnnTrans1 = nn.ConvTranspose1d(self.in_channels, 64, kernel_size=kernel_size)
        h_shape   = ((self.h_shape - 1) + (kernel_size - 1) + 1)

        self.cnnTrans2 = nn.ConvTranspose1d(64, 32, kernel_size=kernel_size)
        self.upsample2 = nn.Upsample(scale_factor=2)
        h_shape   = ((h_shape - 1) + (kernel_size - 1) + 1)*2

        self.cnnTrans3 = nn.ConvTranspose1d(32, out_dim, kernel_size=kernel_size)
        self.upsample3 = nn.Upsample(scale_factor=2)
        h_shape   = ((h_shape - 1) + (kernel_size - 1) + 1)*2
        self.relu      = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = x.reshape(-1,self.in_channels,self.h_shape)
        x = self.relu(self.cnnTrans1(x))
        x = self.relu(self.cnnTrans2(x))
        x = self.upsample2(x)
        x = self.upsample3(self.cnnTrans3(x))
        return x

class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        hidden_dim   = 40
        self.encoder = Encoder(config.window_len, out_dim=hidden_dim, in_dim=config.n_channels)
        self.decoder = Decoder(in_dim=hidden_dim, out_dim=config.n_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

