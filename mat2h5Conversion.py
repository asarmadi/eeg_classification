import h5py
from utils.config import Config
import numpy as np
from utils.utils import spectrogram_per_channel, Normalize

config = Config(None, None)

n_samples_train = len(config.train_subjects)*config.n_windows*config.n_trial*2
n_samples_valid = len(config.valid_subjects)*config.n_windows*config.n_trial*2
n_samples_test  = len(config.test_subjects) *config.n_windows*config.n_trial*2

nTimeBins   = (config.window_len - config.nperseg) // (config.nperseg - config.noverlap) + 1

train_shape = (n_samples_train, config.n_channels, config.freq_cut, nTimeBins)
valid_shape = (n_samples_valid, config.n_channels, config.freq_cut, nTimeBins)
test_shape  = (n_samples_test,  config.n_channels, config.freq_cut, nTimeBins)

f_train = h5py.File(config.file_path+'train_2d.h5', "w")
f_train.create_dataset("data", train_shape)
f_train.create_dataset("label", (n_samples_train,))
f_train.create_dataset("subject", (n_samples_train,))
f_train.create_dataset("trial", (n_samples_train,))

f_test = h5py.File(config.file_path+'test_2d.h5', "w")
f_test.create_dataset("data", test_shape)
f_test.create_dataset("label", (n_samples_test,))
f_test.create_dataset("subject", (n_samples_test,))
f_test.create_dataset("trial", (n_samples_test,))


f_valid = h5py.File(config.file_path+'valid_2d.h5', "w")
f_valid.create_dataset("data", valid_shape)
f_valid.create_dataset("label", (n_samples_valid,))
f_valid.create_dataset("subject", (n_samples_valid,))
f_valid.create_dataset("trial", (n_samples_valid,))

u = 0
s = 0    # sth sample in test set
v = 0    # vth sample in validation set

path = config.file_path + "SF_Obs_MLdata.mat"

f = h5py.File(path,'r')

train_eeg = []
for subject in config.train_subjects:
    for condition in range(config.conditions): # 0, 1 correpond to Flex First, and Extend First
        print(f'subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        train_eeg.append(eeg)
#
train_eeg = np.array(train_eeg)
train_eeg = np.mean(train_eeg, axis=0)
mean, std = np.mean(train_eeg, axis=0), np.std(train_eeg, axis=0)
print(mean.shape)

for subject in config.train_subjects:
    for condition in range(config.conditions): # 0, 1 correpond to Flex First, and Extend First
        print(f'Train subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(config.n_trial):
            for j_windows in range(config.n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:], config)
                f_train["data"][u,...] = Sxx
                f_train["subject"][u]  = subject
                f_train["label"][u]    = condition
                f_train["trial"][u]    = i_trial
                u += 1

for subject in config.test_subjects:
    for condition in range(config.conditions): # 0, 1 correpond to Flex First, and Extend First
        print(f'Test subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(config.n_trial):
            for j_windows in range(config.n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:], config)
                f_test["data"][s,...] = Sxx
                f_test["subject"][s]  = subject
                f_test["label"][s]    = condition
                f_test["trial"][s]    = i_trial
                s += 1

for subject in config.valid_subjects:
    for condition in range(config.conditions): # 0, 1 correpond to Flex First, and Extend First
        print(f'Valid subject#: {subject}, condition: {condition}')
        ref = f["data"][condition][subject]
        eeg = np.array(f[ref])
        eeg = Normalize(mean, std, eeg)
        for i_trial in range(config.n_trial):
            for j_windows in range(config.n_windows):
                Sxx = spectrogram_per_channel(eeg[i_trial,j_windows*config.window_inc:j_windows*config.window_inc+config.window_len,:], config)
                f_valid["data"][v,...] = Sxx
                f_valid["subject"][v]  = subject
                f_valid["label"][v]    = condition
                f_valid["trial"][v]    = i_trial
                v += 1

f_train.close()
f_test.close()
f_valid.close()

