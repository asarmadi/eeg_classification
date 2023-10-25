'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import time
import torch
from models.cnn_model import Net
from models.capsnet import CapsNet
from utils.hdf5_dataset import *
import numpy as np
from scipy import signal


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def test(model, dataloader, model_type, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if model_type == 'cnn':
                 outputs = net(inputs)
            elif model_type == 'capsnet':
                 outputs, reconstructions, masked = model(inputs)
                 onehot_tensor = onehot_encode(targets, device)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if model_type == 'capsnet':
               correct += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(onehot_tensor.data.cpu().numpy(), 1))
            else:
               correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

        print('Test Acc: {0:.3f} ({1}/{2})'.format(100.*correct/total, correct, total))
        return 100.*correct/total

def spectrogram_per_channel(g, config):
    channels = []
    for j in range(config.n_channels):
        f, t, Sxx = signal.stft(g[:, j], fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap)
        channels.append(np.abs(Sxx[:config.freq_cut,:]))
    return np.array(channels)

def Normalize(ave, std, x):
    return ((x-ave)/std)

def onehot_encode(labels, device):
    onehot_tensor = torch.zeros(*labels.shape, 2) # 10 classes for MNIST
    labels = labels.type(torch.LongTensor)
    onehot_tensor = onehot_tensor.scatter(1, labels.view(-1, 1), 1)
    onehot_tensor = onehot_tensor.to(device)
    return onehot_tensor

def model_loader(config):
    if config.model_type == 'cnn':
       return Net(config.in_shape, config.n_subjects)
    elif config.model_type == 'capsnet':
       return CapsNet(config)
    
def data_loader(batch_size, num_workers):
    trainset = HDF5Dataset('./data/train_2d.h5')
    validset = HDF5Dataset('./data/valid_2d.h5')
    testset  = HDF5Dataset('./data/test_2d.h5')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return trainloader, validloader, testloader


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

