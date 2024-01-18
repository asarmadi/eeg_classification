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
import numpy as np
from scipy import signal
import pandas as pd
import tqdm
from stockwell import st
from utils.gaussianTrans import GaussianFourierFeatureTransform


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def test(model, dataloader, config, name_str):
    if config.maj_vote:
       return test_majority_voting(model, dataloader, config, name_str)
    trans_net = None
    if config.transform != "nothing":
       trans_net = trans_loader(config)
       tras_net.eval()

    model.eval()
    correct = 0
    total = 0
    if config.apply_gauss:
       gauss_obj = GaussianFourierFeatureTransform(1, config.mapping_size, 10)
    with torch.no_grad():
        for batch_idx, (inputs, targets, subjects) in enumerate(dataloader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = get_outputs(model, inputs, config, transformer=trans_net)
            _, predicted = outputs.max(1)
            #predicted = outputs.round()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

        print('Acc: {0:.3f} ({1}/{2})'.format(100.*correct/total, correct, total))
        return 100.*correct/total, subjects.unique().numpy()

def test_majority_voting(model, dataloader, config, name_str):
    columns = []
    columns.append('subject')
    columns.append('label')
    columns.append('prediction')
    columns.append('trial')
    columns.append('condition')
    model.eval()
    correct = 0
    total = 0
    trans_net = None
    if config.transform != "nothing":
       trans_net = trans_loader(config)

    results = torch.tensor([]).to(config.device)
    if config.apply_gauss:
       gauss_obj = GaussianFourierFeatureTransform(1, config.mapping_size, 10)
    with torch.no_grad():
        for batch_idx, (inputs, targets, subjects, trials, conditions) in enumerate(dataloader):
            inputs, targets, trials, conditions = inputs.to(config.device), targets.to(config.device), trials.to(config.device), conditions.to(config.device)
            outputs = get_outputs(model, inputs, config, transformer=trans_net)
            _, predicted = outputs.max(1)
            #predicted = outputs.round()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            hh = torch.cat((subjects.reshape(-1,1).to(config.device), targets.reshape(-1,1), predicted.reshape(-1,1),\
                            trials.reshape(-1,1),conditions.reshape(-1,1) ), dim=1)
            results = torch.cat((results, hh))
            progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

        df = pd.DataFrame(results.cpu().numpy(),columns=columns)
        df.to_csv('./csv_out/'+name_str+'_'+str(subjects.unique().numpy()[0])+'.csv', encoding='utf-8', index=False)

        subjects = df['subject'].unique()
        trials = df['trial'].unique()
        labels = df['label'].unique()
        conditions = df['condition'].unique()
        print(subjects)
        print(f"Shapes: S:{len(subjects)}, T:{len(trials)}, L:{len(labels)}, C:{len(conditions)}")
        total = 0
        correct = 0
        with tqdm.tqdm(total=len(df)) as pbar:
           for sub in subjects:
               for tr in trials:
                   for condition in conditions:
                       rows = df[(df['subject'] == sub) & (df['trial'] == tr) & (df['condition'] == condition)]
                       correct_pred = rows[rows['label'] == rows['prediction']]
                       if len(correct_pred) >= config.threshold*len(rows['label']):
                          correct += 1
                       total += 1

    print('Acc: {0:.3f} ({1}/{2})'.format(100.*correct/total, correct, total))
    return 100.*correct/total, subjects


def get_outputs(net, inputs, config, transformer=None):
    if config.apply_gauss:
       inputs = GaussianFourierFeatureTransform(inputs.reshape(-1,1,config.window_len,config.n_channels),config)
#    if config.model_type == 'eegnet' or config.model_type == 'shalloweeg':
 #      inputs = inputs.permute(0,2,1)
    if transformer != None:
       inputs = transformer.get_feature(inputs)
    if config.model_type == 'capsnet':
       outputs, reconstructions, masked = net(inputs)
       outputs = outputs.reshape(-1,config.n_classes)
#       outputs = nn.functional.logsoftmax(outputs,dim=1)
    else:
       outputs = net(inputs.float())
    return outputs

def count_num_classes(dataloader, label):
    n_all_samples, n_label_samples = 0, 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        n_all_samples += targets.size(0)
        n_label_samples += sum(targets == label)
    print(f'Label {label}: {n_label_samples}/{n_all_samples}')

def spectrogram_per_channel(g, config):
    channels = []
    for j in range(config.n_channels):
        f, t, Sxx = signal.stft(g[:, j], fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap)
        channels.append(np.abs(Sxx[:config.freq_cut,:]))
    return np.array(channels)

def stockwell(g, config, single_channel=False):
    channels = []
    fmin_samples = int(config.fmin*config.sig_time)
    fmax_samples = int(config.fmax*config.sig_time)
    if not single_channel:
       for j in config.channels_list:
           stock = st.st(g[:, j], fmin_samples, fmax_samples, gamma=0.1, win_type='kazemi')
           channels.append(np.abs(stock))
       return np.array(channels)
    else:
        stock = st.st(g, fmin_samples, fmax_samples, gamma=0.1, win_type='kazemi')
        return np.abs(stock)

def Normalize(ave, std, x):
    return ((x-ave)/std)

def scale(X):
    min_des, max_des   = -1, 1
    min_curr, max_curr = -332718.2, 29996.145
    #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_std = (X - min_curr) / (max_curr - min_curr)
    X_scaled = X_std * (max_des - min_des) + min_des
    return X_scaled

def preprocess_signal(config, X):
    if config.downsample:
       XD = []
       for i in range(X.shape[1]):
           XD.append(signal.decimate(X[:,i], q=4))
       X = np.array(XD).T
    if config.preprocess_normalize:
       mean, std  = np.mean(X, axis=0), np.std(X, axis=0)
       X = Normalize(mean, std, X)
    if config.preprocess_scale:
       X = scale(X)
    return X


def onehot_encode(labels, device):
    print(labels.shape)
    onehot_tensor = torch.zeros(*labels.shape, 1)
    print(onehot_tensor.shape)
    labels = labels.type(torch.LongTensor)
    print(labels.view(-1, 1).shape)
    onehot_tensor = onehot_tensor.scatter(1, labels.view(-1, 1), 1)
    onehot_tensor = onehot_tensor.to(device)
    return onehot_tensor

def trans_loader(config):
    if config.transform == 'ae':
       from models.ae import AE
       net = AE(config)
    elif config.transform == 'aelinear':
       from models.ae_linear import AELinear
       net = AELinear(config)
    elif config.transform == 'aelstm':
       from models.ae_lstm import AEGRU
       net = AEGRU(config)
    else:
       return None
    net = net.to(config.device)
    net.load_state_dict(torch.load('./checkpoint/Net_'+config.transform+'.pth'))
    net.eval()
    return net


def model_loader(config, kernel_size):
    if config.model_type == 'cnn':
       from models.cnn_model import Net
       return Net(config, kernel_size)
    elif config.model_type == 'capsnet':
       from models.capsnet import CapsNet
       return CapsNet(config)
    elif config.model_type == 'lstm':
       from models.lstm import LSTMClassifier
       return LSTMClassifier(config)
    elif config.model_type == 'cnnlstm':
       from models.cnn_lstm import LSTMConv
       return LSTMConv(config,kernel_size)
    elif config.model_type == 'cnn1d':
       from models.cnn_1d import Conv1D
       return Conv1D(config, kernel_size)
    elif config.model_type == 'eegnet':
       from braindecode.models import EEGNetv4
       return EEGNetv4(in_chans=config.n_channels,n_classes=config.n_classes,input_window_samples=config.window_len,final_conv_length='auto')
    elif config.model_type == 'shalloweeg':
       from braindecode.models import ShallowFBCSPNet
       return ShallowFBCSPNet(in_chans=config.n_channels,n_classes=config.n_classes,input_window_samples=config.window_len,final_conv_length='auto')
    elif config.model_type == 'combined':
       from models.combined import Combined
       return Combined(config,kernel_size)
    elif config.model_type == 'ae':
       from models.ae import AE
       return AE(config)
    elif config.model_type == 'aelinear':
       from models.ae_linear import AELinear
       return AELinear(config)
    elif config.model_type == 'aelstm':
       from models.ae_lstm import AEGRU
       return AEGRU(config)
    elif config.model_type == 'mlp':
       from models.mlp import MLP
       return MLP(config)
    else:
       return False

def data_loader(batch_size, num_workers, transform, valid_check, config, add_trial=False):
    from utils.hdf5_dataset import HDF5Dataset
    if transform == 'stf':
       name_str = '2dstft'
    elif transform == 'stockwell':
       name_str = '2dstfockwell'
    else:
       name_str = '1d'

    if config.apply_valid_set == 'single':
       print(config.subject_idx)
       assert config.subject_idx != "none", "subject should be mentioned in single mode"
       name_str = name_str + '_' + config.subject_idx
    trainset = HDF5Dataset('./data/train_'+name_str+'.h5', config, add_trial=add_trial)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    if config.apply_valid_set == 'single':
       return trainloader, None, None
    if valid_check:
       validset = HDF5Dataset('./data/valid_'+name_str+'.h5', config, add_trial=add_trial)
    testset  = HDF5Dataset('./data/test_'+name_str+'.h5', config, add_trial=add_trial)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    if valid_check:
       validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
       return trainloader, validloader, testloader
    return trainloader, None, testloader


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

