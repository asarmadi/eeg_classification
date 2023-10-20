from nina_helper import *
import csv
import numpy as np
import os
import random
import pandas as pd
import h5py
import re
import scipy 
from config import *
from scipy import signal
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
from utils import get_normalization_params, Normalize

print('test_moves:', test_moves)

def apply_mu(input):
    mu = 2048
    return np.sign(input)*(np.log(1+mu*np.abs(input))/(np.log(1+mu)))

def spectrogram_per_channel(g, s_list):
    channels = []
#    random.shuffle(s_list)
    for j in s_list:
        f, t, Sxx = signal.stft(g[:, j], fs=2000, nperseg=1000, noverlap=950)
        channels.append(np.abs(Sxx[:250,:]))
    return np.array(channels)

n_channels = len(sensors_list)

n_samples_train = n_sub*len(train_reps)*len(train_moves)*((whole_win-window_len)/window_inc+1)
n_samples_valid = n_sub*len(valid_reps)*len(test_moves) *((whole_win-window_len)/window_inc+1)
n_samples_test  = n_sub*len(test_reps) *len(test_moves) *((whole_win-window_len)/window_inc+1)

train_shape = (n_samples_train, n_channels, 250, 25)
valid_shape = (n_samples_valid, n_channels, 250, 25)
test_shape  = (n_samples_test,  n_channels, 250, 25)

f_train = h5py.File(file_path+'train_2d'+name_str+'_'+str(whole_win)+'.h5', "w")
f_train.create_dataset("data", train_shape)
f_train.create_dataset("labels", (n_samples_train,))
f_train.create_dataset("moves", (n_samples_train,))
f_train.create_dataset("reps", (n_samples_train,))

f_test = h5py.File(file_path+'test_2d'+name_str+'_'+str(whole_win)+'.h5', "w")
f_test.create_dataset("data", test_shape)
f_test.create_dataset("labels", (n_samples_test,))
f_test.create_dataset("moves", (n_samples_test,))
f_test.create_dataset("reps", (n_samples_test,))


f_valid = h5py.File(file_path+'valid_2d'+name_str+'_'+str(whole_win)+'.h5', "w")
f_valid.create_dataset("data", valid_shape)
f_valid.create_dataset("labels", (n_samples_valid,))
f_valid.create_dataset("moves", (n_samples_valid,))
f_valid.create_dataset("reps", (n_samples_valid,))

u = 0
s = 0    # sth sample in test set
v = 0    # vth sample in validation set

#f = open('data/nina_stats_min.csv', 'w')
#writer = csv.writer(f)
#f1 = open('data/nina_stats_max.csv', 'w')
#writer1 = csv.writer(f1)

train_emg = []
for subject in range(1,41):
    print('subject#:', subject)
    path = "/data/alireza/human_identification/DB2/DB2_s" + str(subject) + "/DB2_s" + str(subject) + "/"
    data_dict = import_db2(path, subject)
    emg = data_dict['emg']
    emg = apply_mu(emg)
    for j in (train_moves):
        for i in train_reps:
            idxs = np.where((data_dict['move'] == j) & (data_dict['rep'] == i))[0]
            idxs = idxs[0:whole_win]
            train_emg.append(emg[idxs])

train_emg = np.array(train_emg)
train_emg = np.mean(train_emg,axis=0)
mean, std = np.mean(train_emg, axis=0), np.std(train_emg, axis=0)

for subject in range(1,41):
    print('subject#:', subject)
    path = "/data/alireza/human_identification/DB2/DB2_s" + str(subject) + "/DB2_s" + str(subject) + "/"
    data_dict = import_db2(path, subject)
    emg = data_dict['emg']
    emg = apply_mu(emg)
    emg = Normalize(mean,std, emg)

    for j in (train_moves):
        for i in train_reps:
            idxs = np.where((data_dict['move'] == j) & (data_dict['rep'] == i))[0]
            idxs = idxs[0:whole_win]
            possible_targets = np.array(range(idxs[0] + (window_len-1), idxs[-1]+1, window_inc))
            for win_end in possible_targets:
                win_start = win_end - (window_len-1)
                Sxx = spectrogram_per_channel(emg[win_start:win_end+1], sensors_list)
                f_train["data"][u,...] = Sxx
                f_train["moves"][u]    = j
                f_train["labels"][u]   = subject
                f_train["reps"][u]     = i
                u += 1

    for j in (test_moves):
        for i in test_reps:
            idxs = np.where((data_dict['move'] == j) & (data_dict['rep'] == i))[0]
            idxs = idxs[0:whole_win]
            possible_targets = np.array(range(idxs[0] + (window_len-1), idxs[-1]+1, window_inc))
            for win_end in possible_targets:
                win_start = win_end - (window_len-1)
                Sxx = spectrogram_per_channel(emg[win_start:win_end+1],test_sensors_list)
                f_test["data"][s,...] = Sxx
                f_test["moves"][s]    = j
                f_test["labels"][s]   = subject
                f_test["reps"][s]     = i
                s += 1

    for j in (valid_moves):
        for i in valid_reps:
            idxs = np.where((data_dict['move'] == j) & (data_dict['rep'] == i))[0]
            idxs = idxs[0:whole_win]
            possible_targets = np.array(range(idxs[0] + (window_len-1), idxs[-1]+1, window_inc))
            for win_end in possible_targets:
                win_start = win_end - (window_len-1)
                Sxx = spectrogram_per_channel(emg[win_start:win_end+1], test_sensors_list)
                f_valid["data"][v,...] = Sxx
                f_valid["moves"][v]    = j
                f_valid["labels"][v]   = subject
                f_valid["reps"][v]     = i
                v += 1

f_train.close()
f_test.close()
f_valid.close()

